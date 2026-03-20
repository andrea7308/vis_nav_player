import pickle
import os
import json
import networkx as nx
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

def build_topological_graph(json_path, subsample_rate=1):
    """
    Parses exploration JSON into a traversable Directed Graph.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Filter Out Useless Frames
    # We only care about frames where a pure movement action was taken.
    valid_actions = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
    
    filtered_data = [
        step for step in data 
        if len(step['action']) == 1 and step['action'][0] in valid_actions
    ]
            
    # 2. Apply Downsampling
    # Take every Nth frame to space out our waypoints
    sampled_data = filtered_data[::subsample_rate]
    print(f"Reduced {len(data)} total frames to {len(sampled_data)} waypoints.")
    
    # 3. Build the Directed Graph
    G = nx.DiGraph()
    
    # Dictionary to automatically calculate return trip actions
    reverse_actions = {
        'FORWARD': 'BACKWARD', 
        'BACKWARD': 'FORWARD', 
        'LEFT': 'RIGHT', 
        'RIGHT': 'LEFT'
    }
    
    for i in range(len(sampled_data)):
        current_step = sampled_data[i]
        node_id = current_step['step']
        image_file = current_step['image']
        
        # Add the node and store the image filename inside it
        G.add_node(node_id, image=image_file)
        
        # Connect to the previous node to build the sequence
        if i > 0:
            prev_step = sampled_data[i-1]
            prev_node_id = prev_step['step']
            
            # The action taken AT the previous node to get to the current node
            action_taken = prev_step['action'][0] 
            
            # Forward edge: prev_node -> current_node
            G.add_edge(prev_node_id, node_id, action=action_taken, weight=1.0)
            
            # Reverse edge: current_node -> prev_node (for backtracking)
            if action_taken in reverse_actions:
                return_action = reverse_actions[action_taken]
                G.add_edge(node_id, prev_node_id, action=return_action, weight=1.0)
                
    return G, sampled_data


class CNNFeatureExtractor:
    def __init__(self, image_dir="data/images/"):
        self.image_dir = image_dir
        
        # Use GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load pre-trained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        
        # Strip the final classification layer (fc) to get raw features
        # This turns ResNet into a pure feature extractor
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval() # Set to evaluation mode

        # Standard ImageNet preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_feature(self, image_filename: str) -> np.ndarray:
        """Loads an image, runs it through the CNN, and returns a 1D vector."""
        img_path = os.path.join(self.image_dir, image_filename)
        
        # Load image and convert to RGB (drops alpha channel if present)
        img = Image.open(img_path).convert('RGB')
        
        # Preprocess and add batch dimension [1, 3, 224, 224]
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features
            feature = self.model(img_tensor)
            
            # Flatten the output from [1, 512, 1, 1] to a 1D vector [512]
            feature = feature.squeeze().cpu().numpy()
            
            # L2 Normalize the vector so we can easily compare them later
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
                
        return feature
    
def populate_graph_with_features(graph, sampled_data, extractor):
    """
    Iterates through the graph nodes and injects CNN features.
    """
    print("Extracting CNN features for graph nodes...")
    
    # We iterate through sampled_data because it matches our graph nodes
    for step_data in tqdm(sampled_data, desc="Processing Images"):
        node_id = step_data['step']
        image_file = step_data['image']
        
        # Extract the 512-D vector
        feature_vector = extractor.extract_feature(image_file)
        
        # Store it directly in the node attributes
        graph.nodes[node_id]['feature'] = feature_vector
        
    print("Feature extraction complete!")
    return graph

if __name__ == "__main__":
    # Define the path to your json file
    DATA_INFO_PATH = "data/data_info.json"
    
    # 1. Build the topological graph from the JSON
    print("Building map geometry...")
    G, sampled_data = build_topological_graph(DATA_INFO_PATH, subsample_rate=5)
    
    # 2. Initialize the CNN and extract features for all nodes
    cnn_extractor = CNNFeatureExtractor(image_dir="data/images/")
    G = populate_graph_with_features(G, sampled_data, cnn_extractor)
    
    # 3. Save the completed map to the cache directory
    os.makedirs('cache', exist_ok=True)
    with open('cache/topological_map.pkl', 'wb') as f:
        pickle.dump(G, f)
        
    print("Map successfully built and saved to cache/topological_map.pkl!")
