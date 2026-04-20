import os
import cv2
from global_planner import GlobalPlanner

def test_topological_route_step_by_step(current_node, target_node, img_dir='data/exploration_data/images'):
    """
    Calculates the route and displays the corresponding images one by one.
    """
    # 1. Initialize the planner (loads topological_graph.json)
    planner = GlobalPlanner(graph_path='topological_graph.json')
    
    # 2. Get the full path
    max_nodes = planner.G.number_of_nodes()
    path, intents = planner.get_movement_strategy(
        current_node, 
        target_node, 
        lookahead_size=max_nodes
    )
    
    if not path:
        print(f"No path found between Node {current_node} and Node {target_node}.")
        return
        
    print(f"Visualizing route: {path}")
    print("Navigation Controls: Press any key to advance to the next image. Press 'q' to quit.")
    
    # 3. Iterate through the path and display images one at a time
    for i, node_id in enumerate(path):
        # Fetch the image filename from the planner's graph data
        node_data = planner.graph_data.get(node_id, {})
        img_filename = node_data.get('image')
        
        if img_filename:
            img_path = os.path.join(img_dir, img_filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Overlay the current node ID and step number on the image
                text = f"Step {i+1}/{len(path)} | Node: {node_id}"
                cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow("Route Viewer", img)
                
                # Wait indefinitely for a key press
                key = cv2.waitKey(0) & 0xFF
                
                # Allow the user to break out of the loop early
                if key == ord('q'):
                    print("Exiting viewer...")
                    break
            else:
                print(f"[Warning] Image file found but could not be read: {img_path}")
        else:
            print(f"[Warning] No image data mapped for Node {node_id}")
            
    # Clean up the OpenCV window when finished
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage: Replace with your actual start and target node IDs
    START_NODE = 0
    TARGET_NODE = 997
    
    test_topological_route_step_by_step(START_NODE, TARGET_NODE)