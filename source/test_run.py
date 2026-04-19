import os
import cv2
import argparse
import pygame
import vis_nav_game
from vis_nav_game import Player, Action, Phase

# Import your localizer and planner
from visual_localizer import VisualLocalizer
from global_planner import GlobalPlanner

class LocalizerTestPlayer(Player):
    def __init__(self, localizer, planner, target_node, start_node=0, direction="forward"):
        super().__init__()
        self.localizer = localizer
        self.planner = planner
        self.target_node = target_node
        self.last_known_node = start_node
        self.direction = direction
        
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.q_pressed_last_frame = False

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
            pygame.K_ESCAPE: Action.QUIT
        }

    def pre_navigation(self):
        """Called automatically by the engine when Exploration finishes or is skipped."""
        super().pre_navigation()
        
        # Reset actions safely
        self.last_act = Action.IDLE
        self.q_pressed_last_frame = False
        
        # Flush the Pygame event queue to prevent any lingering KEYUP events (like releasing ESC)
        # from accidentally bleeding into the Navigation phase
        pygame.event.clear() 
        
        print("\n" + "="*50)
        print(" EXPLORATION SKIPPED/COMPLETE. STARTING NAVIGATION PHASE.")
        print(f" Target Node: {self.target_node}")
        print(" You have full control. Press 'Q' to localize and plan.")
        print("="*50 + "\n")

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    # If an unmapped key is pressed, show targets
                    self._show_target_images()
                    
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    # Use &= ~ (AND NOT) instead of ^= (XOR) to safely remove 
                    # the bit flag without accidentally toggling it back on
                    self.last_act &= ~self.keymap[event.key]
                    
        return self.last_act

    def _show_target_images(self):
        """Helper to display the goal images if you press an unmapped key."""
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
            
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        cv2.imshow('Target Goals', concat_img)
        cv2.waitKey(1)

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        # Dynamic UI feedback so you aren't guessing what phase the game is in
        if self._state:
            if self._state[1] == Phase.EXPLORATION:
                pygame.display.set_caption("EXPLORATION PLAYBACK - Please Wait...")
            elif self._state[1] == Phase.NAVIGATION:
                pygame.display.set_caption(f"NAVIGATION | Dir: {self.direction} | Press 'Q' to Localize")

                # Localizer logic is strictly isolated to the Navigation Phase
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q] and not self.q_pressed_last_frame:
                    print(f"\n[LOCALIZER] Triggered. Searching near Node {self.last_known_node} (Dir: {self.direction})")
                    
                    current_node = self.localizer.localize(
                        self.fpv, 
                        self.last_known_node, 
                        direction=self.direction
                    )
                    
                    if current_node is not None:
                        print(f"[LOCALIZER] Success! Localized to Node {current_node}")
                        self.last_known_node = current_node
                        
                        # --- PLANNER INTEGRATION ---
                        print(f"[PLANNER] Calculating path to Target Node {self.target_node}...")
                        
                        path, intents = self.planner.get_movement_strategy(
                            current_node=current_node, 
                            target_node=self.target_node, 
                            lookahead_size=20 # Bumped up to 20 per your requirement
                        )
                        
                        if path and len(path) > 1:
                            print("\n>>> UPCOMING SEQUENCE (Next 20 Nodes) <<<")
                            for i in range(min(len(intents), 5)): # Only print first 5 to keep terminal clean
                                print(f"    Step {i+1}: Node {path[i]} -> {path[i+1]} (Intent: {intents[i]})")
                            if len(intents) > 5:
                                print(f"    ... and {len(intents) - 5} more steps.")
                            print("----------------------------------------")
                            
                            # --- NEW DECISION VECTOR ---
                            final_decision = self.planner.distill_directional_intent(intents)
                            print(f"\n>>> IMMEDIATE DIRECTIONAL INTENT VECTOR <<<")
                            print(f"    {final_decision}\n")
                            
                        elif current_node == self.target_node:
                            print("\n>>> TARGET REACHED! <<<\n")
                        else:
                            print(f"\n[PLANNER ERROR] No valid path from {current_node} to {self.target_node}.\n")
                        # ---------------------------

                    else:
                        print("[LOCALIZER] Failed to identify current position.")
                        
                    self.q_pressed_last_frame = True
                    
                elif not keys[pygame.K_q]:
                    self.q_pressed_last_frame = False

        # Render FPV to Pygame window
        rgb = fpv[:, :, ::-1]
        shape = rgb.shape[1::-1]
        pygame_image = pygame.image.frombuffer(rgb.tobytes(), shape, 'RGB')
        self.screen.blit(pygame_image, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Harness for visual_localizer.py and global_planner.py")
    parser.add_argument("--start_node", type=int, default=0, 
                        help="The initial node ID the robot assumes it is starting at (Default: 0)")
    parser.add_argument("--target_node", type=int, required=True, 
                        help="The destination node ID the robot needs to reach")
    parser.add_argument("--direction", choices=["forward", "backward"], default="forward",
                        help="Tells the localizer which way the robot is traveling")
    parser.add_argument("--graph", type=str, default="topological_graph.json", 
                        help="Path to topological graph JSON")
    parser.add_argument("--embeddings", type=str, default="node_embeddings.pt", 
                        help="Path to precomputed embeddings tensor")
    parser.add_argument("--img_dir", type=str, default="data/exploration_data/images", 
                        help="Directory containing exploration images")
    
    args = parser.parse_args()

    if not (os.path.exists(args.graph) and os.path.exists(args.embeddings)):
        print(f"Error: Missing {args.graph} or {args.embeddings}. Run preprocessing.py first.")
        exit(1)

    print("Loading Visual Localizer (CNN weights & SIFT)...")
    localizer = VisualLocalizer(args.graph, args.embeddings, args.img_dir)
    print("Localizer loaded successfully!")
    
    print("\nLoading Global Planner...")
    planner = GlobalPlanner(args.graph)

    print("\nBooting vis_nav_game...")
    player = LocalizerTestPlayer(
        localizer=localizer, 
        planner=planner,
        target_node=args.target_node,
        start_node=args.start_node, 
        direction=args.direction
    )
    
    vis_nav_game.play(the_player=player)