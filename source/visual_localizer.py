import os
import cv2
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Optional imports for PyGame mode
try:
    from vis_nav_game import Player, Action, Phase
    import pygame
except ImportError:
    Player, Action, Phase = object, None, None
    pygame = None

class VisualLocalizer:
    def __init__(self, graph_path, embeddings_path, img_dir, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.img_dir = img_dir
        
        with open(graph_path, 'r') as f:
            raw_graph = json.load(f)
            self.graph = {int(k): v for k, v in raw_graph.items()}
            
        self.total_nodes = len(self.graph)
        self.embeddings = torch.load(embeddings_path, map_location=self.device)
        
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sift = cv2.SIFT_create()
        
        # Build an undirected graph for accurate topological radius searching
        self.undirected_graph = {n: set() for n in self.graph.keys()}
        for node, data in self.graph.items():
            # Add sequential backwards connection manually (since hallways are two-way)
            if node > 0:
                self.undirected_graph[node].add(node - 1)
                self.undirected_graph[node - 1].add(node)
                
            for edge in data.get("edges", []):
                self.undirected_graph[node].add(edge)
                if edge in self.undirected_graph:
                    self.undirected_graph[edge].add(node)
    
    def _get_topological_neighborhood(self, start_node, depth=15):
        """Returns all nodes within 'depth' jumps on the topological graph."""
        visited = set()
        queue = [(start_node, 0)]
        
        while queue:
            curr, d = queue.pop(0)
            if curr not in visited and d <= depth:
                visited.add(curr)
                for neighbor in self.undirected_graph.get(curr, []):
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))
        return list(visited)

    def _get_raw_match_count(self, current_frame_gray, map_img_path):
        """Returns raw number of good SIFT matches without strict RANSAC homography."""
        img2 = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)
        if img2 is None:
            return 0

        kp1, des1 = self.sift.detectAndCompute(current_frame_gray, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return 0

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        return len(good_matches)

    def _verify_match_with_ransac(self, current_frame_gray, map_img_path, min_inliers=25):
        """Strict forward-facing RANSAC verification."""
        img2 = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)
        if img2 is None:
            return False

        kp1, des1 = self.sift.detectAndCompute(current_frame_gray, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return False

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        max_angle_diff = 30.0 
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                angle1 = kp1[m.queryIdx].angle
                angle2 = kp2[m.trainIdx].angle
                diff = abs(angle1 - angle2)
                diff = min(diff, 360.0 - diff) 
                if diff <= max_angle_diff:
                    good_matches.append(m)
                
        if len(good_matches) < 4:
            return False

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is None:
            return False
            
        inlier_count = np.sum(mask)
        if inlier_count < min_inliers:
            return False

        inlier_pts = dst_pts[mask.ravel() == 1]
        if len(inlier_pts) == 0:
            return False
            
        x_coords = inlier_pts[:, 0, 0]
        image_width = img2.shape[1]
        num_bins = 5
        min_active_bins = 3
        min_pts_per_bin = 3
        
        bins = np.linspace(0, image_width, num_bins + 1)
        bin_indices = np.digitize(x_coords, bins) - 1
        
        active_bins = 0
        for b in range(num_bins):
            pts_in_bin = np.sum(bin_indices == b)
            if pts_in_bin >= min_pts_per_bin:
                active_bins += 1
                
        return active_bins >= min_active_bins

    def _extract_feature(self, frame_rgb):
        img = Image.fromarray(frame_rgb)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model(tensor)
            feat = torch.flatten(feat, 1)
            feat = nn.functional.normalize(feat, p=2, dim=1)
            
        return feat

        
    def localize(self, current_frame_bgr, last_known_node, direction="forward", expected_path=None, backward_offset=1):
        """
        Main function to localize the robot.
        expected_path: A list of node IDs representing the planned route towards the target.
        """
        frame_rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
        current_feat = self._extract_feature(frame_rgb)
        similarities = torch.matmul(self.embeddings, current_feat.T).squeeze().cpu().numpy()

        frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
        
        if direction == "forward":
            edges = self.graph.get(last_known_node, {}).get("edges", [])
            temporal_node = last_known_node + 1

            # 1. TEMPORAL INERTIA
            if temporal_node in edges:
                map_img_path = os.path.join(self.img_dir, self.graph[temporal_node]["image"])
                if self._verify_match_with_ransac(frame_gray, map_img_path, min_inliers=15):
                    return temporal_node

            # 2. INTENTIONAL JUMPS
            visual_nodes = [n for n in edges if n != temporal_node]
            visual_nodes = sorted(visual_nodes, key=lambda n: similarities[n], reverse=True)
            
            for node in visual_nodes:
                map_img_path = os.path.join(self.img_dir, self.graph[node]["image"])
                if self._verify_match_with_ransac(frame_gray, map_img_path, min_inliers=25):
                    return node
                    
            # 3. FALLBACK: Search along the planned route
            window_nodes = []
            
            if expected_path and last_known_node in expected_path:
                # We have a planned route to the target. Search along this route.
                curr_idx = expected_path.index(last_known_node)
                start_idx = max(0, curr_idx - 5)
                end_idx = min(len(expected_path), curr_idx + 31)
                
                # Extract nodes from the path, ignoring ones we already checked as edges
                window_nodes = [expected_path[i] for i in range(start_idx, end_idx) if expected_path[i] not in edges]
                
                # Sort by proximity to the current index in the path
                window_nodes = sorted(window_nodes, key=lambda n: abs(expected_path.index(n) - curr_idx))
            else:
                # Failsafe: If no path is provided, fall back to pure numerical node checking
                search_start = max(0, last_known_node - 5)
                search_end = min(self.total_nodes, last_known_node + 31)
                window_nodes = [n for n in range(search_start, search_end) if n not in edges]
                window_nodes = sorted(window_nodes, key=lambda n: abs(n - last_known_node))
            
            for node in window_nodes:
                if similarities[node] < 0.60: 
                    continue 
                map_img_path = os.path.join(self.img_dir, self.graph[node]["image"])
                if self._verify_match_with_ransac(frame_gray, map_img_path, min_inliers=25):
                    print(f"[RECOVERY] Went off course. Recovered at Node {node} along expected path.")
                    return node

            print("\n[ERROR] LOST: Checked expected path and nearby nodes with no valid RANSAC feature matches.")
            return None

        if direction == "backward":
            window_nodes = self._get_topological_neighborhood(last_known_node, depth=15)
            
            best_node = None
            best_sim = -1.0
            
            for node in window_nodes:
                sim = similarities[node]
                if sim > best_sim:
                    best_sim = sim
                    best_node = node
            
            if best_sim > 0.85: 
                print(f"[BACKWARD MODE] Pure CNN Similarity matched Node {best_node} (Sim: {best_sim:.4f}).")
                expected_seq = last_known_node - 1
                if best_node == expected_seq:
                    guessed_node = min(self.total_nodes - 1, max(0, best_node + backward_offset))
                    return guessed_node
                return best_node
            
            print(f"\n[ERROR] BACKWARD LOST: No CNN matches above threshold in node topological radius.")
            return None

# ==============================================================================
# PyGame Integration & Testing Harness
# ==============================================================================
class LocalizerTestPlayer(Player):
    def __init__(self, localizer, start_node=0, direction="forward"):
        super().__init__()
        self.localizer = localizer
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

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption(f"Localizer Test (Press 'Q') | Dir: {self.direction}")

        if self._state and self._state[1] == Phase.NAVIGATION:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q] and not self.q_pressed_last_frame:
                print(f"\n[LOCALIZER] Triggered manually. Last known: {self.last_known_node}, Dir: {self.direction}")
                
                # Note: Test harness doesn't have a planner, so expected_path is left out (falls back to numeric)
                current_node = self.localizer.localize(self.fpv, self.last_known_node, direction=self.direction)
                
                if current_node is not None:
                    print(f"[LOCALIZER] Localized to Node {current_node}")
                    self.last_known_node = current_node
                
                self.q_pressed_last_frame = True
            elif not keys[pygame.K_q]:
                self.q_pressed_last_frame = False

        rgb = fpv[:, :, ::-1]
        shape = rgb.shape[1::-1]
        pygame_image = pygame.image.frombuffer(rgb.tobytes(), shape, 'RGB')
        self.screen.blit(pygame_image, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Visual Localizer")
    parser.add_argument("--mode", choices=["game", "image"], required=True, 
                        help="Run in 'game' mode (Pygame) or 'image' mode (static image testing).")
    parser.add_argument("--image", type=str, 
                        help="Path to the test image (Required if mode is 'image')")
    parser.add_argument("--last_known", type=int, default=0, 
                        help="The last known node ID to start the search from")
    parser.add_argument("--direction", choices=["forward", "backward"], default="forward",
                        help="Simulate the direction of travel")
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

    print("Initializing Visual Localizer...")
    localizer = VisualLocalizer(args.graph, args.embeddings, args.img_dir)

    if args.mode == "image":
        if not args.image or not os.path.exists(args.image):
            print("Error: You must provide a valid --image path when running in 'image' mode.")
            exit(1)
            
        print(f"Testing localizer on static image: {args.image} (Direction: {args.direction})")
        test_frame = cv2.imread(args.image)
        
        if test_frame is not None:
            current_node = localizer.localize(test_frame, args.last_known, direction=args.direction)
            print(f"\nResult: Localized to Node {current_node}")

    elif args.mode == "game":
        if pygame is None:
            print("Error: vis_nav_game and pygame are required to run in 'game' mode.")
            exit(1)
            
        import vis_nav_game as vng
        print("Starting Pygame environment...")
        
        player = LocalizerTestPlayer(localizer, start_node=args.last_known, direction=args.direction)
        vng.play(the_player=player)