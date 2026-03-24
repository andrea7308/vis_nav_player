import torch
import pickle
import numpy as np
import networkx as nx
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from vis_nav_game import Player, Action, Phase
import vis_nav_game
import cv2
import pygame

class CNNAutonomousPlayer(Player):
    def __init__(self, map_path="map_data.pkl"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 1. Load CNN
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Load Map
        with open(map_path, 'rb') as f:
            self.G = pickle.load(f)
        
        # Build a feature matrix for fast cosine similarity lookups
        self.num_nodes = self.G.number_of_nodes()
        # Ensure we are extracting the exact node IDs present in the graph
        self.node_ids = list(self.G.nodes())
        self.map_features = np.zeros((self.num_nodes, 512)) # ResNet18 output size
        
        for idx, node_id in enumerate(self.node_ids):
            self.map_features[idx] = self.G.nodes[node_id]['features']
            
        # Initialize state variables
        self.reset()
        self.waiting_printed = False

    def reset(self):
        """Required by the vis_nav_game engine. Clears state between runs."""
        # Initialize pygame just to keep the OS window manager from freezing
        pygame.init() 
        
        self.fpv = None
        self.current_node = None
        self.target_node = None
        self.expected_next_node = None
        self.action_queue = []
        self.tick_count = 0

    def _extract_features(self, img_array):
        """Converts BGR numpy array (from cv2/game) to CNN features"""
        # The game might pass None if it hasn't fully loaded the view yet
        if img_array is None:
            return np.zeros(512)
            
        img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor).flatten().cpu().numpy()

    def set_target_images(self, images):
        """Game engine hook that gets called when target images are ready."""
        super().set_target_images(images)
        self._find_target_node()

    def pre_navigation(self):
        super().pre_navigation()
        self._find_target_node()

    def _find_target_node(self):
        """Safely attempts to locate the target node."""
        if self.target_node is not None:
            return # Target already found

        targets = self.get_target_images()
        if not targets: # Prevents the 'NoneType' crash!
            return 
        
        best_sim = -1
        for tgt_img in targets:
            if tgt_img is None:
                continue
            tgt_feat = self._extract_features(tgt_img)
            # Cosine similarity against all map nodes
            sims = np.dot(self.map_features, tgt_feat) / (np.linalg.norm(self.map_features, axis=1) * np.linalg.norm(tgt_feat))
            max_idx = np.argmax(sims)
            
            if sims[max_idx] > best_sim:
                best_sim = sims[max_idx]
                self.target_node = self.node_ids[max_idx]
                
        if self.target_node is not None:
            print(f"Target located at node {self.target_node} with similarity {best_sim:.2f}")

    def see(self, fpv):
        self.fpv = fpv
        # CRITICAL: We must pump the OS event loop every frame to prevent freezing!
        if fpv is not None and fpv.size > 0:
            cv2.imshow("CNN Autonomous Navigator", fpv)
            cv2.waitKey(1)

    def act(self):
        # --- OS ANTI-FREEZE ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return Action.QUIT
        # ----------------------

        # --- THE HEARTBEAT ---
        self.tick_count += 1
        if self.tick_count % 60 == 0:
            phase_name = self._state[1] if self._state else "None"
            print(f"[Heartbeat] Engine is alive. Current Phase: {phase_name}")
        # ---------------------

        # 0. Wait for the game engine and camera to initialize
        if self._state is None or self.fpv is None:
            return Action.IDLE

        phase = self._state[1]

        # 1. SKIP EXPLORATION PHASE
        if phase == Phase.EXPLORATION:
            # We must send QUIT (Escape) exactly ONCE to skip the phase. 
            # If we send it twice, it will accidentally close the entire game.
            if not hasattr(self, 'skip_sent'):
                self.skip_sent = True
                print("\n>>> Sending QUIT once to safely skip exploration phase... <<<\n")
                return Action.QUIT
            
            # Stand still while the engine processes the skip and transitions phases
            return Action.IDLE

        # 2. NAVIGATION PHASE
        if phase == Phase.NAVIGATION:
            # 2a. Find Target
            if self.target_node is None:
                self._find_target_node()
                if self.target_node is None:
                    return Action.IDLE

            # 2b. Localize
            if self.current_node is None:
                current_feat = self._extract_features(self.fpv)
                sims = np.dot(self.map_features, current_feat) / (np.linalg.norm(self.map_features, axis=1) * np.linalg.norm(current_feat))
                max_idx = np.argmax(sims)
                self.current_node = self.node_ids[max_idx]
                print(f"Spawned at node {self.current_node} (Sim: {sims[max_idx]:.2f})")
            elif len(self.action_queue) == 0 and self.expected_next_node is not None:
                self.current_node = self.expected_next_node

            # 2c. Plan & Move
            if not self.action_queue:
                if self.current_node == self.target_node:
                    if not getattr(self, 'arrived_printed', False):
                        print("\n*** ARRIVED AT TARGET LOCATION! ***\n")
                        self.arrived_printed = True
                    # In vis_nav_game, QUIT stops the clock and ends the run successfully.
                    return Action.QUIT 
                
                try:
                    UG = self.G.to_undirected()
                    path = nx.shortest_path(UG, source=self.current_node, target=self.target_node)
                    
                    if len(path) > 1:
                        next_node = path[1]
                        self.expected_next_node = next_node
                        
                        if self.G.has_edge(self.current_node, next_node):
                            act_str = self.G.edges[self.current_node, next_node]['action']
                        else:
                            act_str = self.G.edges[next_node, self.current_node]['action']
                            act_str = self._reverse_action(act_str)
                        
                        self._queue_action(act_str)
                    else:
                        return Action.IDLE
                    
                except nx.NetworkXNoPath:
                    if not getattr(self, 'path_error_printed', False):
                        print(f"No path from {self.current_node} to target {self.target_node}!")
                        self.path_error_printed = True
                    return Action.IDLE

            # 2d. Execute Control
            if self.action_queue:
                return self.action_queue.pop(0)
                
        return Action.IDLE


    def _queue_action(self, act_str):
        mapping = {
            'FORWARD': Action.FORWARD,
            'BACKWARD': Action.BACKWARD,
            'LEFT': Action.LEFT,
            'RIGHT': Action.RIGHT
        }
        
        act = mapping.get(act_str, Action.IDLE)
        
        # Only add actual physical movements to the queue! 
        # This prevents the agent from standing still to process "IDLE" frames.
        if act != Action.IDLE:
            self.action_queue.append(act)

    def _reverse_action(self, act_str):
        rev = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        return rev.get(act_str, 'IDLE')

if __name__ == "__main__":
    vis_nav_game.play(the_player=CNNAutonomousPlayer())
