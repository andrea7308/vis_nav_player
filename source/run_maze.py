
from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import pickle
import networkx as nx
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
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


# ---------------------------------------------------------------------------
# VLAD Feature Extraction
# ---------------------------------------------------------------------------

class CNNFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing CNN on: {self.device}")

        # Load pre-trained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        
        # Strip classification layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, img_array: np.ndarray) -> np.ndarray:
        """Converts an OpenCV BGR image array into a 1D feature vector."""
        if img_array is None or img_array.size == 0:
            return np.zeros(512)

        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(img_tensor).squeeze().cpu().numpy()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
        return feature


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class AutonomousPlayer(Player):

    def __init__(self):
        self.fpv = None
        self.screen = None
        self.exploration_quit_sent = False 
        
        # Closed-Loop Navigation trackers
        self.path_calculated = False
        self.waypoint_queue = []         
        self.expected_current_node = None 
        
        # NEW: Tracks how many times we've spun looking for a landmark
        self.scan_spins = 0  
        
        super().__init__()

        self.extractor = CNNFeatureExtractor()
        self.G = None
        self.goal_node = None
        
        self.action_map = {
            'FORWARD': Action.FORWARD,
            'BACKWARD': Action.BACKWARD,
            'LEFT': Action.LEFT,
            'RIGHT': Action.RIGHT
        }

    # --- Game engine hooks ---
    def reset(self):
        self.fpv = None
        self.screen = None
        self.exploration_quit_sent = False
        self.path_calculated = False
        self.waypoint_queue = []
        self.expected_current_node = None
        self.scan_spins = 0
        pygame.init()

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Autonomous Agent FPV")

        rgb = fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def pre_navigation(self):
        super().pre_navigation()
        self._load_map()
        self._setup_goal()

    def act(self):
        pygame.event.clear()

        if not self._state:
            return Action.IDLE

        if self._state[1] == Phase.EXPLORATION:
            if not self.exploration_quit_sent:
                self.exploration_quit_sent = True
                return Action.QUIT
            return Action.IDLE

        if self._state[1] != Phase.NAVIGATION:
            return Action.IDLE

        # --- NAVIGATION PHASE LOGIC ---
        current_feature = self.extractor.extract(self.fpv)

        # 1. REALITY CHECK: Did we hit a wall or get lost?
        if self.path_calculated and self.expected_current_node is not None:
            expected_feature = self.G.nodes[self.expected_current_node].get('feature')
            
            if expected_feature is not None:
                sim = np.dot(current_feature, expected_feature)
                
                # Dropped threshold to 0.55 so it doesn't panic on minor misalignments
                if sim < 0.55:
                    print(f"Reality Check Failed! (Sim: {sim:.2f}). Expected Node {self.expected_current_node}.")
                    print("Illusion broken! Dropping path and turning to clear our view...")
                    self.path_calculated = False
                    self.waypoint_queue = []
                    self.expected_current_node = None
                    
                    # Force a turn! This prevents the robot from continuously ramming the wall.
                    return Action.LEFT

        # 2. Path Planning (Only runs at start, or if a Reality Check fails)
        if not self.path_calculated:
            start_node, sim = self._find_closest_node(current_feature)
            print(f"Localized current position at Node {start_node} (Confidence: {sim:.2f})")

            # CONFIDENCE GATE: Do not trust aliases or bad guesses!
            if sim < 0.80:
                self.scan_spins += 1
                if self.scan_spins >= 4:
                    print("360 scan complete, still lost. Moving forward to find a landmark...")
                    self.scan_spins = 0
                    return Action.FORWARD
                else:
                    print(f"Confidence {sim:.2f} < 0.80. Scanning the area...")
                    return Action.LEFT

            # If we pass the confidence gate, reset spins and proceed!
            self.scan_spins = 0

            if start_node == self.goal_node:
                return Action.CHECKIN

            try:
                path = nx.shortest_path(self.G, source=start_node, target=self.goal_node, weight='weight')
                self.waypoint_queue = []
                
                # Build the queue of actions
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i+1]
                    edge_data = self.G.get_edge_data(u, v)
                    if edge_data and 'action' in edge_data:
                        self.waypoint_queue.append((v, self.action_map[edge_data['action']]))
                
                self.expected_current_node = start_node
                self.path_calculated = True

            except nx.NetworkXNoPath:
                print(f"No path found from {start_node}! Taking a random step to get a better view...")
                return Action.FORWARD

        # 3. Execution: Pop the next step
        if self.waypoint_queue:
            if self.expected_current_node == self.goal_node:
                return Action.CHECKIN

            next_node, action_to_take = self.waypoint_queue.pop(0)
            print(f"Moving to Node {next_node} | Executing: {action_to_take.name}")
            
            # Update our expectation for the NEXT frame
            self.expected_current_node = next_node
            return action_to_take
            
        else:
            if self.expected_current_node == self.goal_node:
                print("Destination reached! Checking in.")
                return Action.CHECKIN
            return Action.IDLE

    # --- Setup & Localization Helpers ---
    def _load_map(self):
        map_path = 'cache/topological_map.pkl'
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Could not find {map_path}.")
        with open(map_path, 'rb') as f:
            self.G = pickle.load(f)

    def _setup_goal(self):
        targets = self.get_target_images()
        if not targets: return
        goal_feature = self.extractor.extract(targets[0])
        self.goal_node, sim = self._find_closest_node(goal_feature)
        print(f"Goal localized at Node {self.goal_node} (Confidence: {sim:.2f})")

    def _find_closest_node(self, target_feature):
        best_node = None
        max_sim = -1.0
        for node_id, data in self.G.nodes(data=True):
            if 'feature' in data:
                sim = np.dot(target_feature, data['feature'])
                if sim > max_sim:
                    max_sim = sim
                    best_node = node_id
        return best_node, max_sim

if __name__ == "__main__":
    import vis_nav_game
    
    # Start the game with our new Autonomous Player!
    vis_nav_game.play(the_player=AutonomousPlayer())
