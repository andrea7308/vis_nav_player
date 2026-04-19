# local_controller.py
import math
from enum import Enum, auto
from visual_localizer import check_facing_forward


class State(Enum):
    CENTERING  = auto()   # nudging away from walls before committing to forward
    FORWARD    = auto()   # driving toward the next node
    PRE_TURN   = auto()   # a turn is coming — creep forward to the turn point
    TURNING    = auto()   # executing the rotation in place
    RECOVERING = auto()   # off_course flag raised — hold and spin to relocalize
    REORIENTING = auto()


# ── Tuning constants ─────────────────────────────────────────────────────────
# All speeds are units robot driver understands (e.g. -1.0 to 1.0)

FORWARD_SPEED       = 0.4    # normal cruise speed
CREEP_SPEED         = 0.2    # slow approach when near a turn point
TURN_SPEED          = 0.35   # rotation speed (in-place)
CENTERING_SPEED     = 0.15   # lateral nudge speed

WALL_DANGER_THRESH  = 0.20   # if centroid offset > this fraction of frame width → re-centre
WALL_CLEAR_THRESH   = 0.10   # once offset drops below this → done centering
TURN_APPROACH_NODES = 3      # how many nodes ahead counts as "turn coming soon"
RECOVER_SPIN_TICKS  = 20     # how many ticks to spin before giving up and snapping


class LocalController:
    """
    Finite-state machine that converts:
      - a directional intent  (from global_planner)
      - a traversable-area map (from perception_stabilizer)
    into concrete motor commands every tick.

    Motor commands are dicts:
        {"forward": float, "turn": float}
    where forward ∈ [-1, 1] and turn ∈ [-1, 1].
    Positive turn = clockwise / right.
    """

    def __init__(self):
        self.state = State.CENTERING
        self._recover_ticks = 0
        self._turn_direction = 0      # +1 right, -1 left, set when entering PRE_TURN
        self._turn_remaining = 0.0    # degrees left to rotate, decremented each tick

    def update(
        self,
        direction: str,          # "FORWARD" | "LEFT" | "RIGHT" | "UTURN" | "ARRIVED"
        traversable_map: dict,   # output of perception_stabilizer
        off_course: bool,        # from visual_localizer
        lookahead: list[str],    # next N directions from global_planner
        current_frame_path: str,
        next_node_path: str,
    ) -> dict:
        """
        Run one FSM tick. Returns a motor command dict.

        Args:
            direction:       The immediate direction the global_planner wants.
            traversable_map: {
                "centroid_offset": float,   # -1=hard left, +1=hard right, 0=centred
                "left_space":  float,       # free pixels left of centre (normalised 0-1)
                "right_space": float,       # free pixels right of centre
                "floor_visible": bool       # is there any floor at all?
            }
            off_course:      True if visual_localizer lost confidence.
            lookahead:       List of upcoming directions (index 0 = next after current).
        """
        facing_forward = check_facing_forward(current_frame_path, next_node_path)
        if not facing_forward and direction == "FORWARD":
            self.state = State.REORIENTING
            self._turn_remaining = 180.0
            self._turn_direction = 1
            return self._stop()

        # ── Global interrupt: lost localisation ──────────────────────────────
        if off_course and self.state != State.RECOVERING:
            self._enter_recovering()

        # ── State machine ────────────────────────────────────────────────────
        if self.state == State.RECOVERING:
            return self._tick_recovering(off_course)

        if self.state == State.CENTERING:
            return self._tick_centering(traversable_map, direction, lookahead)

        if self.state == State.FORWARD:
            return self._tick_forward(traversable_map, direction, lookahead, off_course)

        if self.state == State.PRE_TURN:
            return self._tick_pre_turn(traversable_map, direction)

        if self.state == State.TURNING:
            return self._tick_turning()

        return self._stop()

    # ── State entry helpers ───────────────────────────────────────────────────

    def _enter_recovering(self):
        self.state = State.RECOVERING
        self._recover_ticks = 0

    def _enter_pre_turn(self, direction: str):
        self.state = State.PRE_TURN
        self._turn_direction = 1 if direction in ("RIGHT", "UTURN") else -1
        self._turn_remaining = 90.0 if direction != "UTURN" else 180.0

    def _enter_turning(self):
        self.state = State.TURNING

    def _tick_recovering(self, off_course: bool) -> dict:
        """
        Spin slowly in place so the localizer gets a fresh look at the walls.
        After RECOVER_SPIN_TICKS ticks, give up spinning and go back to CENTERING
        (the localizer will have snapped to its best guess by then).
        """
        self._recover_ticks += 1
        if not off_course or self._recover_ticks >= RECOVER_SPIN_TICKS:
            self.state = State.CENTERING
            return self._stop()
        return {"forward": 0.0, "turn": TURN_SPEED * 0.5}

    def _tick_centering(
        self, tmap: dict, direction: str, lookahead: list[str]
    ) -> dict:
        """
        Nudge laterally until the centroid is close enough to centre,
        then transition to FORWARD (or straight to PRE_TURN if a turn is imminent).
        """
        offset = tmap.get("centroid_offset", 0.0)

        if abs(offset) <= WALL_CLEAR_THRESH:
            # Centred — decide what to do next
            if self._turn_is_imminent(direction, lookahead):
                self._enter_pre_turn(direction)
            else:
                self.state = State.FORWARD
            return self._stop()

        # Nudge toward centre: positive offset = drifted right → turn left (negative)
        nudge = -math.copysign(CENTERING_SPEED, offset)
        return {"forward": 0.0, "turn": nudge}

    def _tick_forward(
        self, tmap: dict, direction: str, lookahead: list[str], off_course: bool
    ) -> dict:
        """
        Drive forward. Continuously monitor:
          - Wall proximity → drop back to CENTERING if needed
          - Upcoming turn in lookahead → switch to PRE_TURN
          - Arrival → stop
        Also applies a soft steering correction based on centroid offset
        so the robot naturally tracks the corridor centre while moving.
        """
        if direction == "ARRIVED":
            self.state = State.CENTERING
            return self._stop()

        offset = tmap.get("centroid_offset", 0.0)

        # Wall danger check
        if abs(offset) > WALL_DANGER_THRESH:
            self.state = State.CENTERING
            return self._stop()

        # Upcoming turn check
        if self._turn_is_imminent(direction, lookahead):
            self._enter_pre_turn(direction)
            return self._stop()

        # Normal forward with gentle corridor-centering correction
        # The correction is proportional to offset — small drift = small nudge
        correction = -offset * 0.3   # damped: full offset → 30% of max turn
        correction = max(-0.4, min(0.4, correction))  # clamp

        return {"forward": FORWARD_SPEED, "turn": correction}

    def _tick_pre_turn(self, tmap: dict, direction: str) -> dict:
        """
        Creep forward slowly until the floor ahead narrows significantly
        (meaning we've reached the corner / junction), then execute the turn.
        
        Use left_space / right_space from perception_stabilizer:
        - turning right: right_space drops sharply when the right wall closes in
        - turning left: left_space drops sharply
        """
        left  = tmap.get("left_space", 1.0)
        right = tmap.get("right_space", 1.0)

        at_turn_point = False
        if self._turn_direction > 0 and right < 0.25:
            at_turn_point = True
        elif self._turn_direction < 0 and left < 0.25:
            at_turn_point = True

        if at_turn_point:
            self._enter_turning()
            return self._stop()

        return {"forward": CREEP_SPEED, "turn": 0.0}

    def _tick_turning(self) -> dict:
        """
        Rotate in place by decrementing _turn_remaining each tick.
        When done, go back to CENTERING (which re-checks walls before driving).
        """
        DEGREES_PER_TICK = 5.0   # tune this to match your robot's turn speed

        self._turn_remaining -= DEGREES_PER_TICK
        if self._turn_remaining <= 0:
            self.state = State.CENTERING
            return self._stop()

        return {"forward": 0.0, "turn": TURN_SPEED * self._turn_direction}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _turn_is_imminent(self, direction: str, lookahead: list[str]) -> bool:
        """
        A turn is "imminent" if the current direction IS a turn, or if one
        appears within the next TURN_APPROACH_NODES steps of the lookahead.
        """
        if direction in ("LEFT", "RIGHT", "UTURN"):
            return True
        for upcoming in lookahead[:TURN_APPROACH_NODES]:
            if upcoming in ("LEFT", "RIGHT", "UTURN"):
                return True
        return False

    @staticmethod
    def _stop() -> dict:
        return {"forward": 0.0, "turn": 0.0}