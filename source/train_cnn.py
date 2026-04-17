import json
import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import cv2
import numpy as np

def load_and_subsample_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 1. Filter out steps where the action is strictly ["IDLE"]
    active_steps = [entry for entry in data if "IDLE" not in entry.get('action', [])]
    
    # 2. Subsample: Take 1 out of every 8 remaining images
    subsampled_data = active_steps[::8]
    
    print(f"Original steps: {len(data)}")
    print(f"Active steps: {len(active_steps)}")
    print(f"Subsampled steps: {len(subsampled_data)}")
    
    return subsampled_data

def verify_match_with_ransac(img1_path, img2_path, min_inliers=25, min_width_ratio=0.25):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return False

    # Initialize SIFT
    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return False

    # L2 Norm for SIFT
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's Ratio Test & Orientation Constraint
    good_matches = []
    
    # Maximum allowed rotation between features (in degrees)
    # 30 degrees allows for perspective shifts while turning, but blocks upside-down matches
    max_angle_diff = 30.0 
    
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            
            # Extract the angles of the matching keypoints
            angle1 = kp1[m.queryIdx].angle
            angle2 = kp2[m.trainIdx].angle
            
            # Calculate the absolute difference, accounting for the 360-degree wrap-around
            # e.g., matching 355 degrees and 5 degrees is only a 10-degree difference
            diff = abs(angle1 - angle2)
            diff = min(diff, 360.0 - diff) 
            
            # Only keep the match if the feature is upright relative to its pair
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

    # --- HORIZONTAL SPREAD & CLUSTER CHECK (SPATIAL BINNING) ---
    inlier_pts = dst_pts[mask.ravel() == 1]
    
    if len(inlier_pts) == 0:
        return False
        
    # Isolate the X coordinates
    x_coords = inlier_pts[:, 0, 0]
    image_width = img2.shape[1]
    
    # Configuration for Spatial Binning
    num_bins = 5             # Divide the image into 5 vertical columns
    min_active_bins = 3      # Require matches to be present in at least 3 of those columns
    min_pts_per_bin = 3      # Require at least 3 points in a column for it to be considered "active"
    
    # Create bin boundaries
    bins = np.linspace(0, image_width, num_bins + 1)
    
    # Assign each x-coordinate to a bin (digitize returns 1-indexed bins, subtract 1 for 0-index)
    bin_indices = np.digitize(x_coords, bins) - 1
    
    # Count how many bins meet our cluster threshold
    active_bins = 0
    for b in range(num_bins):
        pts_in_bin = np.sum(bin_indices == b)
        if pts_in_bin >= min_pts_per_bin:
            active_bins += 1
            
    # Reject if the matches aren't sufficiently distributed in clusters across the image
    if active_bins < min_active_bins:
        return False
        
    return True

def mine_loop_closures(json_path, img_dir, subsample_rate=8, time_margin=20, similarity_threshold=0.92):
    # 1. Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    active_steps = [entry for entry in data if "IDLE" not in entry.get('action', [])]
    subsampled_data = active_steps[::subsample_rate]
    
    # This checks for Apple Silicon (MPS), then NVIDIA (CUDA), then falls back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 2. Load a standard pre-trained ResNet as a feature extractor
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove the final classification layer to get raw features
    model = nn.Sequential(*list(model.children())[:-1]).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Extracting baseline features...")
    embeddings = []
    
    with torch.no_grad():
        for item in tqdm(subsampled_data):
            img_path = os.path.join(img_dir, item['image'])
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract features and flatten
            feat = model(tensor)
            feat = torch.flatten(feat, 1)
            # L2 Normalize
            feat = nn.functional.normalize(feat, p=2, dim=1)
            embeddings.append(feat)
            
    embeddings = torch.cat(embeddings, dim=0) # Shape: [N, 512]
    
    print("Computing similarity matrix...")
    # Matrix multiplication of normalized vectors gives cosine similarity between all pairs
    sim_matrix = torch.matmul(embeddings, embeddings.T) 
    
    shortcuts = {}
    
    # 3. Find pairs that are visually similar but far apart in time
    window_size = 1 # Check 1 frame forward and 1 frame back
    
    for i in range(len(subsampled_data)):
        shortcuts[i] = []
        for j in range(len(subsampled_data)):
            
            if abs(i - j) > time_margin:
                # STAGE 1: Temporal Sequence CNN Candidate Generation
                seq_match = True
                
                # Verify the sequence of frames [i-1, i, i+1] matches [j-1, j, j+1]
                for w in range(-window_size, window_size + 1):
                    w_i, w_j = i + w, j + w
                    
                    # Boundary check
                    if 0 <= w_i < len(subsampled_data) and 0 <= w_j < len(subsampled_data):
                        # If any frame in the sequence drops below similarity, reject the whole sequence
                        if sim_matrix[w_i, w_j] < 0.85: 
                            seq_match = False
                            break
                    else:
                        seq_match = False
                        break
                        
                if seq_match and sim_matrix[i, j] > 0.88:
                    # STAGE 2: Geometric Verification
                    img1_path = os.path.join(img_dir, subsampled_data[i]['image'])
                    img2_path = os.path.join(img_dir, subsampled_data[j]['image'])
                    
                    if verify_match_with_ransac(img1_path, img2_path, min_inliers=25, min_width_ratio=0.25):
                        shortcuts[i].append(j)
                
    # Save the shortcuts map
    with open("visual_shortcuts.json", "w") as f:
        json.dump(shortcuts, f)
        
    print("Found and saved visual shortcuts!")
    return shortcuts

class MazeTripletDatasetWithShortcuts(Dataset):
    def __init__(self, data_list, img_dir, shortcuts_dict, transform=None, time_margin=10):
        self.data_list = data_list
        self.img_dir = img_dir
        self.transform = transform
        self.time_margin = time_margin
        self.shortcuts = shortcuts_dict

    def __len__(self):
        return len(self.data_list) - 1

    def __getitem__(self, idx):
        anchor_info = self.data_list[idx]
        
        # Determine Positive
        str_idx = str(idx) # JSON keys become strings
        has_shortcuts = str_idx in self.shortcuts and len(self.shortcuts[str_idx]) > 0
        
        # 30% chance to use a visual shortcut (if one exists), 70% chance to use temporal neighbor
        if has_shortcuts and random.random() < 0.3:
            positive_idx = random.choice(self.shortcuts[str_idx])
            positive_info = self.data_list[positive_idx]
        else:
            positive_info = self.data_list[idx + 1]

        # Determine Negative (must be outside time margin AND not a visual shortcut)
        negative_idx = random.randint(0, len(self.data_list) - 1)
        
        # Keep searching if it's too close in time OR if it's actually a known shortcut
        while (abs(negative_idx - idx) <= self.time_margin) or \
              (has_shortcuts and negative_idx in self.shortcuts[str_idx]):
            negative_idx = random.randint(0, len(self.data_list) - 1)
            
        negative_info = self.data_list[negative_idx]

        # Load and transform images...
        anchor_img = Image.open(os.path.join(self.img_dir, anchor_info['image'])).convert('RGB')
        positive_img = Image.open(os.path.join(self.img_dir, positive_info['image'])).convert('RGB')
        negative_img = Image.open(os.path.join(self.img_dir, negative_info['image'])).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img
    

class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ResNetEmbedding, self).__init__()
        # Load a pre-trained ResNet18 (good balance of speed/power for navigation)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Isolate the backbone by removing the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a new linear layer to project to our desired embedding dimension
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # L2 Normalize embeddings to project them onto a unit sphere
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
    
def train_maze_model():
    # Setup paths
    json_path = 'data/exploration_data/data_info.json'
    img_dir = 'data/exploration_data/images'
    shortcuts_path = 'visual_shortcuts.json' # <-- ADD THIS
    
    # Preprocess
    subsampled_data = load_and_subsample_data(json_path)
    
    # Load the shortcuts dictionary we mined earlier <-- ADD THIS
    with open(shortcuts_path, 'r') as f:
        shortcuts_dict = json.load(f)
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create Dataset & DataLoader <-- UPDATE THIS
    dataset = MazeTripletDatasetWithShortcuts(
        data_list=subsampled_data, 
        img_dir=img_dir, 
        shortcuts_dict=shortcuts_dict, # <-- Pass the dictionary here
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize Model, Loss, and Optimizer
    # This checks for Apple Silicon (MPS), then NVIDIA (CUDA), then falls back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ResNetEmbedding(embedding_dim=128).to(device)
    
    # The margin pushes negative samples to be at least 'margin' distance further 
    # from the anchor than the positive sample is.
    criterion = nn.TripletMarginLoss(margin=1.0, p=2) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 10
    
    print(f"Starting training on {device}...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass all three images through the same network weights (Siamese network style)
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            
            # Calculate loss
            loss = criterion(emb_a, emb_p, emb_n)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save the trained weights
    torch.save(model.state_dict(), "maze_resnet_embedder.pth")
    print("Training complete and model saved!")

if __name__ == "__main__":
    json_path = 'data/exploration_data/data_info.json'
    img_dir = 'data/exploration_data/images'
    shortcuts_path = 'visual_shortcuts.json'

    # Step 1: Check if the shortcuts file exists. 
    # If it doesn't, run the mining function to create it.
    import os
    if not os.path.exists(shortcuts_path):
        print("Shortcuts file not found. Mining loop closures first... (This may take a few minutes)")
        mine_loop_closures(json_path, img_dir)
        ## False Positives:
        ## Idx 498 - Idx 1127
        ## Idx 579 - Idx 936 and reverse
        ## Idx 779 - Idx 995 and reverse
    else:
        print("Shortcuts file found! Skipping mining phase.")

    # Step 2: Now that the file definitely exists, run the training!
    #print("Starting model training...")
    #train_maze_model()
