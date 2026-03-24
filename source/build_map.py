import os
import json
import torch
import pickle
import networkx as nx
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from tqdm import tqdm

# Adjust this if your script is not running in the parent directory of 'data'
DATA_DIR = "exploration_data" 
OUTPUT_MAP = "map_data.pkl"

class MapBuilder:
    def __init__(self):
        # Load a pre-trained ResNet18 and remove the final classification layer
        print("Loading CNN...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1]).to(self.device)
        self.model.eval()

        # Standard ImageNet normalization for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, img_path):
        """Passes a single image through the CNN to get a 512-d feature vector."""
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor).flatten().cpu().numpy()
        return features

    def build(self):
        G = nx.DiGraph()
        
        # Point directly to the root info file and images folder
        info_path = os.path.join(DATA_DIR, 'data_info.json')
        img_dir = os.path.join(DATA_DIR, 'images')
        
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Could not find {info_path}. Check your DATA_DIR path.")
            
        with open(info_path, 'r') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} frames from {info_path}")
        print("Extracting features and building graph...")
        
        previous_node_id = None
        
        # Iterate through the flat JSON list
        for i in tqdm(range(len(data))):
            frame = data[i]
            img_name = frame['image']
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found. Skipping frame {frame['step']}.")
                continue
            
            # Extract CNN features
            features = self.extract_features(img_path)
            
            # Use the exact step number from the JSON as the node ID
            node_id = frame['step']
            
            # Add node to graph
            G.add_node(node_id, features=features, image=img_name)
            
            # Add edge from previous node based on the action taken to get here
            if previous_node_id is not None:
                # The action at i-1 is what caused the transition to i
                prev_action = data[i-1]['action'][0]
                
                # Connect EVERY frame to guarantee a continuous, unbroken path
                G.add_edge(previous_node_id, node_id, action=prev_action)
            
            previous_node_id = node_id

        with open(OUTPUT_MAP, 'wb') as f:
            pickle.dump(G, f)
            
        print(f"Map saved to {OUTPUT_MAP} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

if __name__ == "__main__":
    builder = MapBuilder()
    builder.build()
