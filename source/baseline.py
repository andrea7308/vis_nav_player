"""
Baseline Level-1: VLAD-based visual navigation.

Pipeline: RootSIFT → KMeans codebook → VLAD → cosine similarity graph → Dijkstra
"""

from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import pickle
import networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import clip
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = "cache"
IMAGE_DIR = "data/images/"
DATA_INFO_PATH = "data/data_info.json"

# Graph construction
TEMPORAL_WEIGHT = 1.0       # edge weight for consecutive frames
VISUAL_WEIGHT_BASE = 2.0    # base weight for visual shortcut edges
VISUAL_WEIGHT_SCALE = 3.0   # weight += scale * vlad_distance
MIN_SHORTCUT_GAP = 50       # minimum trajectory index gap for shortcuts

os.makedirs(CACHE_DIR, exist_ok=True)

class CLIPExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.prev_node = None

    def extract(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(x)

        feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize
        return feat.cpu().numpy().flatten()

    def extract_batch(self, file_list):
        feats = []
        for fname in tqdm(file_list, desc="CLIP"):
            img = cv2.imread(os.path.join(IMAGE_DIR, fname))
            feats.append(self.extract(img))
        return np.array(feats)

# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):

    def __init__(self, n_clusters: int = 128, subsample_rate: int = 5,
                 top_k_shortcuts: int = 30):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super().__init__()
        self.prev_node = None
        self.autonomous = True
        self.current_path = []
        self.path_idx = 0
        self.last_nodes = []
        self.recovery_mode = False
        self.recovery_steps = 0

        self.subsample_rate = subsample_rate
        self.top_k_shortcuts = top_k_shortcuts

        # Load trajectory data
        self.motion_frames = []
        self.file_list = []
        if os.path.exists(DATA_INFO_PATH):
            with open(DATA_INFO_PATH) as f:
                raw = json.load(f)
            pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
            all_motion = [
                {'step': d['step'], 'image': d['image'], 'action': d['action'][0]}
                for d in raw
                if len(d['action']) == 1 and d['action'][0] in pure
            ]
            self.motion_frames = all_motion[::subsample_rate]
            self.file_list = [m['image'] for m in self.motion_frames]
            print(f"Frames: {len(all_motion)} total, "
                  f"{len(self.motion_frames)} after {subsample_rate}x subsample")

        # self.extractor = VLADExtractor(n_clusters=n_clusters)
        self.extractor = CLIPExtractor(device="cpu")
        self.database = None
        self.G = None
        self.goal_node = None

    # --- Game engine hooks ---
    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def act(self):
        # --- AUTONOMOUS MODE ---
        if self.autonomous and self.database is not None and self.goal_node is not None:
            cur = self._get_current_node()

            # --- STUCK DETECTION ---
            self.last_nodes.append(cur)
            if len(self.last_nodes) > 10:
                self.last_nodes.pop(0)

            if len(set(self.last_nodes)) == 1:
                print("[AUTO] STUCK → entering recovery mode")
                self.recovery_mode = True
                self.recovery_steps = 0

            # --- RECOVERY MODE ---
            if self.recovery_mode:
                self.recovery_steps += 1

                if self.recovery_steps > 6:
                    print("[RECOVERY] exiting")
                    self.recovery_mode = False
                else:
                    if self.recovery_steps % 2 == 1:
                        print("[RECOVERY] turning")
                        return Action.LEFT
                    else:
                        print("[RECOVERY] moving forward")
                        return Action.FORWARD

            # --- REPLAN IF NEEDED ---
            if not self.current_path or self.path_idx >= len(self.current_path) or cur != self.current_path[self.path_idx]:
                self.current_path = self._get_path(cur)
                self.path_idx = 0

            # --- CHECK GOAL ---
            if len(self.current_path) <= 1:
                print("Reached goal → CHECKIN")
                return Action.CHECKIN

            # --- FIND NEXT TEMPORAL STEP (avoid going backward) ---
            next_node = None

            for n in self.current_path[1:]:
                if abs(n - cur) == 1:
                    # avoid going back to previous node
                    if self.prev_node is not None and n == self.prev_node:
                        continue
                    next_node = n
                    break

            if next_node is None:
                print("[AUTO] No temporal step → exploring")

                # try to move forward instead of freezing
                return Action.FORWARD

            # --- ACTION ---
            action_str = self._edge_action(cur, next_node)

            if action_str == 'BACKWARD':
                action_str = 'FORWARD'

            action_map = {
                'FORWARD': Action.FORWARD,
                'BACKWARD': Action.BACKWARD,
                'LEFT': Action.LEFT,
                'RIGHT': Action.RIGHT
            }

            action = action_map.get(action_str, Action.IDLE)

            self.prev_node = cur

            print(f"[AUTO] {cur} → {next_node} | action: {action_str}")

            self.path_idx += 1
            return action

        # --- MANUAL MODE (fallback) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        return self.last_act

    def see(self, fpv):
        print("STATE:", self._state)
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption("KeyboardPlayer:fpv")


        if self.database is not None and self.G is not None and self.goal_node is not None:
            self.display_next_best_view()

        rgb = fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    def pre_navigation(self):
        super().pre_navigation()
        self._build_database()
        self._build_graph()
        self._setup_goal()


    def _build_database(self):
        if self.database is not None:
            print("Database already computed, skipping.")
            return

        print("Extracting CLIP features...")
        self.database = self.extractor.extract_batch(self.file_list)
        print(f"Database: {self.database.shape}")

    # --- Navigation graph ---
    def _build_graph(self):
        """Build graph with temporal + visual shortcut edges."""
        if self.G is not None:
            print("Graph already built, skipping.")
            return

        n = len(self.database)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        # Temporal edges (consecutive frames)
        for i in range(n - 1):
            self.G.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT, edge_type="temporal")

        # --- Visual edges: k-NN graph (CLIP-friendly) ---
        print("Computing similarity matrix...")
        sim = self.database @ self.database.T
        np.fill_diagonal(sim, -1)

        k = 10  # try 5–20
        print(f"Building k-NN graph (k={k})...")

        edge_count = 0

        for i in range(n):
            sim_i = sim[i].copy()

            # remove temporal neighbors
            lo = max(0, i - MIN_SHORTCUT_GAP)
            hi = min(n, i + MIN_SHORTCUT_GAP + 1)
            sim_i[lo:hi] = -1

            # top-k neighbors
            nn_idx = np.argpartition(sim_i, -k)[-k:]

            for j in nn_idx:
                s = float(sim_i[j])

                if s <= 0:  # skip weak matches
                    continue

                d = float(np.sqrt(max(0, 2 - 2 * s)))

                self.G.add_edge(
                    i, j,
                    weight=1.0 + 2.0 * d,
                    edge_type="visual"
                )
                edge_count += 1

        print(f"Added ~{edge_count} visual edges")

    # --- Goal ---

    def _setup_goal(self):
        """Set goal node from front-view target image."""
        if self.goal_node is not None:
            print("Goal already set, skipping.")
            return
        targets = self.get_target_images()
        if not targets:
            return
        feat = self.extractor.extract(targets[0])
        sims = self.database @ feat

        k = 3
        topk = np.argpartition(sims, -k)[-k:]
        self.goal_node = int(np.median(topk))
        
        d = float(np.sqrt(max(0, 2 - 2 * sims[self.goal_node])))
        print(f"Goal: node {self.goal_node} (d={d:.4f})")

    # --- Helpers ---
    def _load_img(self, idx: int) -> np.ndarray | None:
        """Load image by database index."""
        if 0 <= idx < len(self.file_list):
            return cv2.imread(os.path.join(IMAGE_DIR, self.file_list[idx]))
        return None
    
    def _get_current_node(self) -> int:
        feat = self.extractor.extract(self.fpv)
        sims = self.database @ feat

        # take top-k instead of single best
        k = 5
        topk = np.argpartition(sims, -k)[-k:]

        node = int(np.median(topk))

        if self.prev_node is not None:
            if abs(node - self.prev_node) > 50:
                node = self.prev_node  # reject jump

        self.prev_node = node
        return node
    
    def _get_path(self, start: int) -> list[int]:
        """Shortest path with goal-directed weighting."""
        try:
            def edge_cost(u, v, d):
                base = d["weight"]

                if d["edge_type"] == "visual":
                    base += 5.0  # penalize visual edges heavily

                # similarity to goal (higher = better)
                sim_v = float(self.database[v] @ self.database[self.goal_node])

                # convert to penalty (lower cost if closer to goal)
                goal_bias = 1.0 - sim_v  # in [0, 2]

                return base + 0.5 * goal_bias  # tune 0.3–1.0

            return nx.shortest_path(
                self.G,
                start,
                self.goal_node,
                weight=edge_cost
            )
        except nx.NetworkXNoPath:
            return [start]

    def _edge_action(self, a: int, b: int) -> str:
        REVERSE = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD',
               'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        
        if b == a + 1 and a < len(self.motion_frames):
            return self.motion_frames[a]['action']
        elif b == a - 1 and b < len(self.motion_frames):
            return REVERSE.get(self.motion_frames[b]['action'], '?')

        # --- NEW: handle visual edges ---
        if b > a:
            return 'FORWARD'
        else:
            return 'BACKWARD'

    # --- Display ---
    def show_target_images(self):
        targets = self.get_target_images()
        if not targets:
            return
        top = cv2.hconcat(targets[:2])
        bot = cv2.hconcat(targets[2:])
        img = cv2.vconcat([top, bot])
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, pos in [('Front', (10, 25)), ('Right', (w//2+10, 25)),
                           ('Back', (10, h//2+25)), ('Left', (w//2+10, h//2+25))]:
            cv2.putText(img, label, pos, font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Target Images', img)
        cv2.waitKey(1)

    def display_next_best_view(self):
        """
        Navigation panel:
            Info bar: current node | goal | hops | next action
            Row 1:    [Live FPV] [Best match] [Target (front)]
            Row 2:    Path preview (next 5 nodes)
        """
        ACT = {'FORWARD': 'FWD', 'BACKWARD': 'BACK', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT'}
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        TW, TH = 260, 195          # main thumbnails
        PW, PH = TW * 3 // 5, TH * 3 // 5   # path preview thumbnails
        N_PREVIEW = 5

        # Localize & plan
        cur = self._get_current_node()
        cur_sim = float(self.database[cur] @ self.extractor.extract(self.fpv))
        cur_d = float(np.sqrt(max(0, 2 - 2 * cur_sim)))
        path = self._get_path(cur)
        hops = len(path) - 1

        # Analyze edges
        edge_info = []
        for a, b in zip(path[:-1], path[1:]):
            et = self.G[a][b].get("edge_type", "temporal")
            if et == "temporal":
                act = ACT.get(self._edge_action(a, b), '?')
                edge_info.append(("seq", act, b == a + 1))
            else:
                edge_info.append(("vis", None, None))
        t_steps = sum(1 for e in edge_info if e[0] == "seq")
        v_jumps = len(edge_info) - t_steps

        if edge_info:
            etype, act, _ = edge_info[0]
            hint = act if etype == "seq" else "VISUAL JUMP"
        else:
            hint = "AT GOAL"
        near = hops <= 5

        # --- Info bar ---
        panel_w = TW * 3
        bar = np.zeros((40, panel_w, 3), dtype=np.uint8)
        bar[:] = (0, 0, 160) if near else (50, 35, 15)
        txt = (f"Node {cur} (d={cur_d:.3f})"
               f"  |  Goal {self.goal_node}"
               f"  |  {hops} hops ({t_steps}s+{v_jumps}v)"
               f"  |  >> {hint}")
        cv2.putText(bar, txt, (8, 27), FONT, 0.48, (255, 255, 255), 1, AA)
        if near:
            cv2.putText(bar, "NEAR TARGET — SPACE",
                        (panel_w - 220, 27), FONT, 0.48, (0, 255, 255), 1, AA)

        # --- Row 1: [FPV] [Match] [Target] ---
        def thumb(img, label, color, extra=None):
            t = cv2.resize(img, (TW, TH))
            cv2.rectangle(t, (0, 0), (TW-1, TH-1), color, 2)
            cv2.putText(t, label, (6, 22), FONT, 0.55, color, 1, AA)
            if extra:
                cv2.putText(t, extra, (6, 44), FONT, 0.45, (200, 200, 200), 1, AA)
            return t

        fpv_t = thumb(self.fpv, "Live FPV", (255, 255, 255))
        match_img = self._load_img(cur)
        if match_img is None:
            match_img = np.zeros((TH, TW, 3), dtype=np.uint8)
        match_t = thumb(match_img, f"Match: node {cur}", (0, 255, 0), f"d={cur_d:.3f}")
        targets = self.get_target_images()
        tgt = targets[0] if targets else np.zeros((TH, TW, 3), dtype=np.uint8)
        tgt_t = thumb(tgt, "Target (front)", (0, 140, 255))
        row1 = cv2.hconcat([fpv_t, match_t, tgt_t])

        # --- Row 2: path preview ---
        preview = path[1:1 + N_PREVIEW]
        cells = []
        for p in range(N_PREVIEW):
            if p < len(preview):
                img = self._load_img(preview[p])
                if img is None:
                    img = np.zeros((PH, PW, 3), dtype=np.uint8)
                img = cv2.resize(img, (PW, PH))
                etype, act, is_fwd = edge_info[p]
                if etype == "seq":
                    lbl = f"{'>' if is_fwd else '<'} {act}"
                    clr = (200, 200, 0)
                else:
                    lbl = "~ VISUAL"
                    clr = (200, 100, 255)
                cv2.rectangle(img, (0, 0), (PW-1, PH-1), clr, 1)
                cv2.putText(img, f"+{p+1} node {preview[p]}", (4, 16),
                            FONT, 0.38, (255, 255, 255), 1, AA)
                cv2.putText(img, lbl, (4, 34), FONT, 0.38, clr, 1, AA)
            else:
                img = np.zeros((PH, PW, 3), dtype=np.uint8)
            cells.append(img)
        row2 = cv2.hconcat(cells)

        # Pad row2 to match panel width
        if row2.shape[1] < panel_w:
            pad = np.zeros((PH, panel_w - row2.shape[1], 3), dtype=np.uint8)
            row2 = cv2.hconcat([row2, pad])

        panel = cv2.vconcat([bar, row1, row2])
        cv2.imshow("Navigation", panel)
        cv2.waitKey(1)
        print(f"Node {cur} -> Goal {self.goal_node} | "
              f"{hops} hops ({t_steps}s+{v_jumps}v) | >> {hint}")


if __name__ == "__main__":
    import argparse
    import vis_nav_game

    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=5,
                        help="Take every Nth motion frame (default: 5)")
    parser.add_argument("--n-clusters", type=int, default=128,
                        help="VLAD codebook size (default: 128)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of global visual shortcut edges (default: 30)")
    args = parser.parse_args()

    vis_nav_game.play(the_player=KeyboardPlayerPyGame(
        n_clusters=args.n_clusters,
        subsample_rate=args.subsample,
        top_k_shortcuts=args.top_k,
    ))
