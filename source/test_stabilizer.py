import os
import random
import cv2
import numpy as np

def test_color_lidar(img_path, num_rays=15):
    # 1. Load Image
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Failed to load {img_path}")
        return

    height, width, _ = bgr.shape
    
    # ==========================================
    # 2. EXACT COLOR MASKING
    # ==========================================
    # OpenCV uses Blue, Green, Red (not RGB)
    white_bgr = np.array([239, 239, 239], dtype=np.uint8)
    blueish_bgr = np.array([224, 186, 163], dtype=np.uint8)
    
    # Adding a tiny color tolerance in case Pygame applies 
    # any anti-aliasing or slight lighting gradients.
    # Set tolerance to 0 if the colors are mathematically perfectly flat.
    tolerance = 8
    
    lower_white = np.clip(white_bgr - tolerance, 0, 255)
    upper_white = np.clip(white_bgr + tolerance, 0, 255)
    
    lower_blue = np.clip(blueish_bgr - tolerance, 0, 255)
    upper_blue = np.clip(blueish_bgr + tolerance, 0, 255)
    
    # Create masks for the two floor colors
    mask_white = cv2.inRange(bgr, lower_white, upper_white)
    mask_blue = cv2.inRange(bgr, lower_blue, upper_blue)
    
    # Combine them: This mask is 255 (white) where the floor is, 0 (black) elsewhere
    floor_mask = cv2.bitwise_or(mask_white, mask_blue)

    # A 7x7 kernel is usually large enough to bridge anti-aliased seams.
    # If the cracks survive, increase this to (9, 9) or (11, 11).
    close_kernel = np.ones((7, 7), np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, close_kernel)
    
    # Morphological Opening (Optional but recommended)
    # This removes tiny isolated dots of noise AFTER the seams are stitched.
    open_kernel = np.ones((3, 3), np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, open_kernel)
    
    # Invert the mask: Now the WALLS are 255 (white), and the FLOOR is 0 (black)
    obstacle_mask = cv2.bitwise_not(floor_mask)


    # ==========================================
    # 3. HIGH-DENSITY PSEUDO-LIDAR
    # ==========================================
    horizon_y = int(height * 0.4) # Don't look higher than the top 40%
    
    # Generate evenly spaced X coordinates across the screen width
    # We pad the edges slightly (e.g., 5% in) so we don't raycast the literal edge of the screen
    pad = int(width * 0.05)
    ray_x_coords = np.linspace(pad, width - pad, num_rays, dtype=int)
    
    left_free_space = 0
    right_free_space = 0
    hits = []
    
    center_index = num_rays // 2
    
    for i, x in enumerate(ray_x_coords):
        hit_y = horizon_y
        
        # Scan UP the column
        for y in range(height - 1, horizon_y, -1):
            if obstacle_mask[y, x] > 0: # We hit a non-floor pixel!
                hit_y = y
                break
                
        # Calculate how many pixels of floor we traversed before hitting the wall
        ray_distance = (height - 1) - hit_y
        hits.append((x, hit_y))
        
        # Accumulate space for steering
        if i < center_index:
            left_free_space += ray_distance
        elif i > center_index:
            right_free_space += ray_distance
        # Note: If num_rays is odd, we ignore the dead-center ray for the steering balance

    # ==========================================
    # 4. STEERING LOGIC
    # ==========================================
    total_space = left_free_space + right_free_space
    repel_turn = 0.0
    if total_space > 0:
        # Positive = Turn Right, Negative = Turn Left
        repel_turn = (right_free_space - left_free_space) / total_space


    # ==========================================
    # VISUALIZATION 
    # ==========================================
    vis = bgr.copy()
    
    # Draw horizon line
    cv2.line(vis, (0, horizon_y), (width, horizon_y), (0, 255, 255), 1)
    
    # Draw all the rays
    for x, hit_y in hits:
        # Green line for the free space (floor)
        cv2.line(vis, (x, height), (x, hit_y), (0, 255, 0), 1)
        # Red dot where it impacted an obstacle
        cv2.circle(vis, (x, hit_y), 4, (0, 0, 255), -1)
    
    # Draw a dividing line down the middle to show Left vs Right
    cv2.line(vis, (width//2, height), (width//2, horizon_y), (255, 0, 0), 1)
    
    # Print data
    direction = "CENTER"
    if repel_turn > 0.1: direction = "RIGHT"
    elif repel_turn < -0.1: direction = "LEFT"
    
    cv2.putText(vis, f"Turn: {repel_turn:.2f} ({direction})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(vis, f"Rays: {num_rays}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Show the pipeline
    cv2.imshow("1. Semantic LiDAR Array", vis)
    cv2.imshow("2. Floor Mask (White=Floor)", floor_mask)
    cv2.imshow("3. Obstacle Mask (White=Wall)", obstacle_mask)
    
    print(f"Image: {os.path.basename(img_path)} | Turn: {repel_turn:.2f}")

if __name__ == "__main__":
    img_dir = 'data/exploration_data/images'
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    if not all_images:
        print(f"No images found in {img_dir}")
        exit()
        
    print("Press 'w' / 's' to increase/decrease the number of rays.")
    print("Press any other key for the next image. Press 'q' or 'ESC' to quit.")
    
    current_rays = 15
    
    while True:
        random_img = random.choice(all_images)
        test_color_lidar(random_img, num_rays=current_rays)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('w'):
            current_rays += 2
            print(f"Increased rays to {current_rays}")
            # Rerun on the same image to see the difference
            test_color_lidar(random_img, num_rays=current_rays)
            cv2.waitKey(0)
        elif key == ord('s'):
            current_rays = max(3, current_rays - 2)
            print(f"Decreased rays to {current_rays}")
            test_color_lidar(random_img, num_rays=current_rays)
            cv2.waitKey(0)
            
    cv2.destroyAllWindows()