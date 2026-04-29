import cv2
import numpy as np
import locate_walls_2

def verify_match_with_ransac(kp1, kp2, good_matches, img_width, min_spread_ratio=0.0):
    """
    Verifies matches using RANSAC and enforces a horizontal spread check
    to filter out localized false positives.
    """
    if len(good_matches) < 4:
        return None, None
    
    # Extract coordinates of the good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Calculate the horizontal spread of the matched features
    x_coords = src_pts[:, 0, 0]
    horizontal_spread = np.max(x_coords) - np.min(x_coords)
    min_spread = img_width * min_spread_ratio
    
    # Reject the match if the features are too clustered together
    # if horizontal_spread < min_spread:
    #     print(f"Match rejected: Horizontal spread ({horizontal_spread:.1f}px) is below threshold ({min_spread:.1f}px).")
    #     return None, None
        
    # Find Homography from Image 2 to Image 1
    # We map 2 to 1 so we can project Image 2's corners onto Image 1
    H, inlier_mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    return H, inlier_mask

def estimate_pose(img_path1, img_path2):
    # 1. Load Images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None or img2 is None:
        print("Error: Could not load one or both images.")
        return
        
    h1, w1 = img1.shape[:2]
        
    # 2. Get Wall Masks (Isolates features to walls only)
    print("Generating wall masks...")
    _, obs_mask1 = locate_walls_2.create_wall_mask(img_path1)
    _, obs_mask2 = locate_walls_2.create_wall_mask(img_path2)
    
    if obs_mask1 is None or obs_mask2 is None:
        print("Error generating masks.")
        return

    # 3. Feature Detection (SIFT)
    print("Extracting features...")
    sift = cv2.SIFT_create()
    
    # Apply the mask so we only detect keypoints on the walls, ignoring floor/sky
    kp1, des1 = sift.detectAndCompute(img1, mask=obs_mask1)
    kp2, des2 = sift.detectAndCompute(img2, mask=obs_mask2)
    
    # 4. Feature Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    print(f"Found {len(good_matches)} good feature matches.")
            
    # 5. RANSAC Verification & Homography Estimation
    H, mask = verify_match_with_ransac(kp1, kp2, good_matches, img_width=w1)
    
    vis_img = img1.copy()
    
    if H is not None:
        print("Homography successful. Estimating relative pose...")
        
        # Get dimensions of the second image
        h2, w2 = img2.shape[:2]
        
        # Define the 4 corners of Image 2's frame
        img2_corners = np.float32([
            [0, 0], 
            [0, h2 - 1], 
            [w2 - 1, h2 - 1], 
            [w2 - 1, 0]
        ]).reshape(-1, 1, 2)
        
        # 6. Project Image 2's frame onto Image 1
        transformed_corners = cv2.perspectiveTransform(img2_corners, H)
        
        # Draw the estimated FOV polygon
        vis_img = cv2.polylines(vis_img, [np.int32(transformed_corners)], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        
        # Label the projection
        text_x = int(transformed_corners[0][0][0])
        text_y = int(transformed_corners[0][0][1]) - 10
        cv2.putText(vis_img, "Img2 Estimated FOV", (max(0, text_x), max(20, text_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        # Optional: Show the feature matches side-by-side
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask.ravel().tolist(), flags=2)
        match_vis = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
        
        cv2.imshow("Inlier Wall Matches", match_vis)
    else:
        print("Failed to establish a valid homography matrix or insufficient horizontal spread.")
        
    cv2.imshow("Pose Estimation (Img2 Projected onto Img1)", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Point these paths to two sequential or overlapping frames in your dataset
    # img_path_A = "data/exploration_data/images/17650.jpg"
    # img_path_B = "data/exploration_data/images/0.jpg"
    img_path_A = "data/exploration_data/images/18161.jpg"
    img_path_B = "data/exploration_data/images/109.jpg"
    
    estimate_pose(img_path_A, img_path_B)