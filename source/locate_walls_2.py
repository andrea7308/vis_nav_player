import os
import random
import cv2
import numpy as np


def create_wall_mask(img_path, blur_size=5, close_size=7, open_size=3):
    # 1. Load Image
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Failed to load {img_path}")
        return None, None

    # ==========================================
    # NEW: TUNABLE PRE-SMOOTHING
    # ==========================================
    # Applying a Gaussian blur before thresholding cleans up sensor noise 
    # and anti-aliasing artifacts, creating solid, contiguous masks.
    if blur_size > 0:
        # OpenCV requires the blur kernel size to be an odd number greater than 0
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)

    height, width, _ = bgr.shape
    
    # ==========================================
    # EXACT COLOR MASKING
    # ==========================================
    white_bgr = np.array([239, 239, 239])
    blueish_bgr = np.array([224, 186, 163])
    sky = np.array([255, 255, 255])

    tolerance = 8

    lower_white = np.clip(white_bgr - tolerance, 0, 255).astype(np.uint8)
    upper_white = np.clip(white_bgr + tolerance, 0, 255).astype(np.uint8)

    lower_blue = np.clip(blueish_bgr - tolerance, 0, 255).astype(np.uint8)
    upper_blue = np.clip(blueish_bgr + tolerance, 0, 255).astype(np.uint8)

    lower_sky = np.clip(sky - tolerance, 0, 255).astype(np.uint8)
    upper_sky = np.clip(sky + tolerance, 0, 255).astype(np.uint8)
    
    mask_white = cv2.inRange(bgr, lower_white, upper_white)
    mask_blue = cv2.inRange(bgr, lower_blue, upper_blue)
    mask_sky = cv2.inRange(bgr, lower_sky, upper_sky)
    
    floor_mask = mask_white | mask_blue | mask_sky

    # ==========================================
    # TUNABLE MORPHOLOGICAL SMOOTHING
    # ==========================================
    if close_size > 0:
        close_kernel = np.ones((close_size, close_size), np.uint8)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, close_kernel)
    
    if open_size > 0:
        open_kernel = np.ones((open_size, open_size), np.uint8)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, open_kernel)
    
    obstacle_mask = cv2.bitwise_not(floor_mask)

    return floor_mask, obstacle_mask




def test_color_lidar(img_path, num_rays=15, blur_size=5, close_size=7, open_size=3):
    # 1. Load Image
    bgr = cv2.imread(img_path)
    if bgr is None:
        return

    height, width, _ = bgr.shape
    
    # Pass the new tunable parameters into the mask creator
    floor_mask, obstacle_mask = create_wall_mask(img_path, blur_size, close_size, open_size)
    if floor_mask is None:
        return
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
    cv2.imshow("Semantic LiDAR Array", vis)
    cv2.imshow("Floor/Sky Mask (White=Floor/Sky)", floor_mask)
    cv2.imshow("Obstacle Mask (White=Wall)", obstacle_mask)

    wall_segment_vis = draw_wall_segments(bgr, obstacle_mask)
    cv2.imshow("Wall Segments", wall_segment_vis)
    
    print(f"Image: {os.path.basename(img_path)} | Turn: {repel_turn:.2f}")

def extract_wall_polygons(obstacle_mask, jump_threshold=15, min_area_ratio=0.04, epsilon_factor=0.02, smooth_window=15, slope_window=11, slope_threshold=0.20, top_gap_threshold=6, coplanar_slope_tol=0.25):
    """
    Finds individual wall faces by severing the mask at depth discontinuities and corners.
    Includes a post-processing pass to merge falsely split coplanar walls.
    """
    height, width = obstacle_mask.shape
    working_mask = obstacle_mask.copy()
    
    # ==========================================
    # 1. TRACE TOP & BOTTOM PROFILES
    # ==========================================
    bottom_y_coords = np.full(width, height - 1)
    top_y_coords = np.full(width, height - 1) # Default to bottom of screen if no wall
    
    for x in range(width):
        col = working_mask[:, x]
        y_indices = np.nonzero(col)[0]
        if len(y_indices) > 0:
            top_y_coords[x] = y_indices[0]
            bottom_y_coords[x] = y_indices[-1]
            
    # ==========================================
    # 2. FIND SPLIT POINTS (JUMPS & CORNERS)
    # ==========================================
    split_x_indices = set()
    
    # A. Depth Jumps
    for x in range(1, width):
        if abs(bottom_y_coords[x] - bottom_y_coords[x-1]) > jump_threshold:
            split_x_indices.add(x)

    # B. Corners (Changes in bottom slope)
    padded = np.pad(bottom_y_coords, (smooth_window // 2, smooth_window // 2), mode='edge')
    smoothed_y = np.convolve(padded, np.ones(smooth_window)/smooth_window, mode='valid')
    
    slope_diffs = np.zeros(width)
    for x in range(slope_window, width - slope_window):
        y_left = smoothed_y[x - slope_window]
        y_center = smoothed_y[x]
        y_right = smoothed_y[x + slope_window]
        
        slope_left = (y_center - y_left) / slope_window
        slope_right = (y_right - y_center) / slope_window
        slope_diffs[x] = abs(slope_right - slope_left)

    for x in range(slope_window, width - slope_window):
        if slope_diffs[x] > slope_threshold:
            l_bound = max(0, x - slope_window // 2)
            r_bound = min(width, x + slope_window // 2 + 1)
            if slope_diffs[x] == np.max(slope_diffs[l_bound:r_bound]):
                split_x_indices.add(x)

    # ==========================================
    # 3. SEVER & EXTRACT INITIAL MASKS
    # ==========================================
    for x in split_x_indices:
        cv2.line(working_mask, (x, 0), (x, height), 0, 3) # Cut the mask

    contours, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (height * width) * min_area_ratio
    
    initial_masks = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            m = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(m, [cnt], -1, 255, -1)
            
            # Get bounding box for left-to-right sorting
            x, y, w, h = cv2.boundingRect(cnt)
            initial_masks.append({'mask': m, 'min_x': x, 'max_x': x + w})
            
    initial_masks.sort(key=lambda d: d['min_x'])

    # ==========================================
    # 4. NEW: MERGE FALSE SPLITS (COPLANAR CHECK)
    # ==========================================
    merged = True
    while merged:
        merged = False
        for i in range(len(initial_masks) - 1):
            left_wall = initial_masks[i]
            right_wall = initial_masks[i+1]
            
            # 4a. Are they touching/adjacent? (Within 15px to account for the slice line)
            if right_wall['min_x'] - left_wall['max_x'] > 15:
                continue
                
            boundary_x = (left_wall['max_x'] + right_wall['min_x']) // 2
            check_left = max(0, boundary_x - 5)
            check_right = min(width - 1, boundary_x + 5)
            
            # 4b. Is there a gap in the TOP wall line?
            if top_y_coords[check_left] == height - 1 or top_y_coords[check_right] == height - 1:
                continue # Gap to the floor
            if abs(top_y_coords[check_left] - top_y_coords[check_right]) > top_gap_threshold:
                continue # Major height drop
                
            # 4c. Are they on the SAME PLANE?
            # We measure the slope of the TOP edge, because the bottom edge is currently corrupted by going out of frame
            w_left = max(1, left_wall['max_x'] - left_wall['min_x'])
            w_right = max(1, right_wall['max_x'] - right_wall['min_x'])
            
            slope_left = (top_y_coords[left_wall['max_x'] - 1] - top_y_coords[left_wall['min_x']]) / w_left
            slope_right = (top_y_coords[right_wall['max_x'] - 1] - top_y_coords[right_wall['min_x']]) / w_right
            
            if abs(slope_left - slope_right) < coplanar_slope_tol:
                # Zip the masks together
                new_mask = cv2.bitwise_or(left_wall['mask'], right_wall['mask'])
                
                # Perform a horizontal morphological close to bridge the black slice line
                kernel = np.ones((1, 15), np.uint8) 
                new_mask = cv2.dilate(new_mask, kernel, iterations=1)
                new_mask = cv2.erode(new_mask, kernel, iterations=1)
                
                # Recalculate bounding box
                coords = np.column_stack(np.where(new_mask > 0))
                min_x = coords[:, 1].min()
                max_x = coords[:, 1].max()
                
                # Replace the two split masks with the new merged one
                initial_masks.pop(i+1)
                initial_masks.pop(i)
                initial_masks.insert(i, {'mask': new_mask, 'min_x': min_x, 'max_x': max_x})
                
                merged = True
                break # Restart the evaluation loop with the newly combined mask

    # ==========================================
    # 5. FINAL GEOMETRIC APPROXIMATION
    # ==========================================
    wall_masks = []
    wall_polygons = []
    
    for item in initial_masks:
        m = item['mask']
        merged_contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not merged_contours: continue
        
        # Grab the bounding contour of the fully merged shape
        cnt = max(merged_contours, key=cv2.contourArea) 
        
        # Snap to geometric shapes (rectangles/rhombuses)
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
        
        final_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(final_mask, [approx_poly], -1, 255, -1)
        
        wall_masks.append(final_mask)
        wall_polygons.append(approx_poly)
        
    return wall_masks, wall_polygons


def draw_wall_segments(bgr_img, obstacle_mask, alpha=0.8):
    """
    Extracts individual polygonal wall faces and overlays them 
    with distinct transparent colors.
    """
    # Get the isolated masks and polygons
    wall_masks, wall_polygons = extract_wall_polygons(obstacle_mask)
    
    overlay = bgr_img.copy()
    
    # Distinct colors for each wall face
    colors = [
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 0),    # Green
        (255, 128, 0)   # Orange
    ]
    
    for i, poly in enumerate(wall_polygons):
        color = colors[i % len(colors)]
        
        # Draw the filled polygon on the overlay
        cv2.fillPoly(overlay, [poly], color)
        
        # Optional: Draw a crisp border around the rhombus/rectangle to make it pop
        cv2.polylines(overlay, [poly], True, (255, 255, 255), 2)
        
    # Blend the geometric overlay with the original image
    vis_img = cv2.addWeighted(overlay, alpha, bgr_img, 1 - alpha, 0)
    
    return vis_img

if __name__ == "__main__":
    img_dir = 'data/exploration_data/images'
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    if not all_images:
        print(f"No images found in {img_dir}")
        exit()
        
    print("CONTROLS:")
    print("  'w' / 's' : Increase/Decrease Rays")
    print("  'a' / 'd' : Increase/Decrease Image Blur Smoothing")
    print("  'SPACE'   : Next Image")
    print("  'q' / ESC : Quit")
    
    # Default parameters
    current_rays = 15
    current_blur = 2
    current_close = 7
    current_open = 3
    
    while True:
        random_img = random.choice(all_images)
        # random_img = "data/exploration_data/images/17650.jpg"
        # random_img = "data/exploration_data/images/0.jpg"
        # random_img = "data/exploration_data/images/10206.jpg"
        
        while True: # Inner loop to keep reloading the same image while tuning
            test_color_lidar(random_img, num_rays=current_rays, blur_size=current_blur, close_size=current_close, open_size=current_open)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                exit()
            elif key == ord('w'):
                current_rays += 2
                print(f"Rays: {current_rays}")
            elif key == ord('s'):
                current_rays = max(3, current_rays - 2)
                print(f"Rays: {current_rays}")
            elif key == ord('a'):
                current_blur += 2
                print(f"Blur Size: {current_blur}")
            elif key == ord('d'):
                current_blur = max(0, current_blur - 2)
                print(f"Blur Size: {current_blur}")
            else:
                # Any other key (like SPACE) breaks out of the tuning loop to fetch the next image
                break
            
    cv2.destroyAllWindows()