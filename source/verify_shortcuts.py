import json
import os
import matplotlib.pyplot as plt
from PIL import Image

def verify_shortcuts(json_path, img_dir, shortcuts_path="visual_shortcuts.json", subsample_rate=8, samples_to_show=50):
    # 1. Load the original data to get filenames
    with open(json_path, 'r') as f:
        data = json.load(f)
    active_steps = [entry for entry in data if "IDLE" not in entry.get('action', [])]
    subsampled_data = active_steps[::subsample_rate]

    # 2. Load the saved shortcuts
    with open(shortcuts_path, 'r') as f:
        shortcuts = json.load(f)

    # 3. Filter out images that didn't have any matches
    valid_keys = [k for k, v in shortcuts.items() if len(v) > 0]
    
    if not valid_keys:
        print("No shortcuts were found in the JSON file.")
        return

    print(f"Found {len(valid_keys)} images with shortcuts. Displaying {samples_to_show} examples...")

    # 4. Plot a few examples side-by-side
    for key in valid_keys[:samples_to_show]:
        anchor_idx = int(key)
        match_indices = shortcuts[key]
        
        anchor_filename = subsampled_data[anchor_idx]['image']
        
        # Create a plot with enough columns for the Anchor + all its matches
        fig, axes = plt.subplots(1, len(match_indices) + 1, figsize=(4 * (len(match_indices) + 1), 4))
        
        # Handle case where there is exactly 1 match (axes is a 1D array)
        if len(match_indices) == 1:
            axes = [axes[0], axes[1]]

        # Show Anchor
        anchor_img = Image.open(os.path.join(img_dir, anchor_filename)).convert('RGB')
        axes[0].imshow(anchor_img)
        axes[0].set_title(f"Anchor (Idx {anchor_idx})", fontweight="bold")
        axes[0].axis('off')

        # Show Matches
        for i, match_idx in enumerate(match_indices):
            match_filename = subsampled_data[match_idx]['image']
            match_img = Image.open(os.path.join(img_dir, match_filename)).convert('RGB')
            axes[i+1].imshow(match_img)
            axes[i+1].set_title(f"Match (Idx {match_idx})")
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    verify_shortcuts('data/exploration_data/data_info.json', 'data/exploration_data/images')