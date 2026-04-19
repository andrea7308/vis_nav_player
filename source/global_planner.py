import json
import networkx as nx

class GlobalPlanner:
    def __init__(self, graph_path='topological_graph.json'):
        """Initializes the planner and builds the topological graph in memory."""
        print(f"Loading topological graph from {graph_path}...")
        with open(graph_path, 'r') as f:
            self.graph_data = {int(k): v for k, v in json.load(f).items()}
            
        self.G = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """Constructs the NetworkX graph with temporal and visual edges."""
        self.G.add_nodes_from(self.graph_data.keys())

        for node_id, data in self.graph_data.items():
            for edge_target in data.get('edges', []):
                if edge_target == node_id + 1:
                    self.G.add_edge(node_id, edge_target, weight=1.0, edge_type="temporal")
                else:
                    self.G.add_edge(node_id, edge_target, weight=1.5, edge_type="visual")
                    
        print(f"Graph loaded successfully! Nodes: {self.G.number_of_nodes()}, Edges: {self.G.number_of_edges()}")

    def get_movement_strategy(self, current_node, target_node, lookahead_size=20):
        """
        Calculates the shortest path, extracts a lookahead sequence, and translates
        it into a directional intent strategy ensuring the robot faces its direction of travel.
        """
        if current_node == target_node:
            return [current_node], ["ARRIVED"]

        try:
            full_path = nx.shortest_path(self.G, source=current_node, target=target_node, weight="weight")
        except nx.NetworkXNoPath:
            print(f"[Planner Error] No path found between {current_node} and {target_node}")
            return [], []

        # Extract the lookahead window (+1 to get the actual target nodes for the edges)
        lookahead_path = full_path[:lookahead_size + 1] 
        
        # Translate the topological path into actionable intents
        raw_intents = self._translate_to_strategy(lookahead_path)
        
        # --- Optimize kinematics to ensure forward-facing travel ---
        strategy_intents = self._optimize_kinematics(raw_intents)

        return lookahead_path, strategy_intents

    def _translate_to_strategy(self, path):
        """
        Translates raw topological traversal into a forward-facing movement strategy.
        Converts reverse sequential traversal into a chassis reorientation + FORWARD commands.
        """
        strategy = []
        
        # Tracks relative flow: 1 for N->N+1 (forward in dataset), -1 for N->N-1 (reversed)
        current_flow = 1 

        for i in range(len(path) - 1):
            curr = path[i]
            nxt = path[i+1]
            step_intent = ""

            # 1. Traversing sequentially forward along exploration data
            if nxt == curr + 1:
                if current_flow == -1:
                    step_intent += "REORIENT_TO_FORWARD_FLOW + "
                    current_flow = 1
                    
                action = self.graph_data[curr].get('action', ['IDLE'])[0]
                step_intent += action
                strategy.append(step_intent)

            # 2. Traversing sequentially backward along exploration data
            elif nxt == curr - 1:
                if current_flow == 1:
                    step_intent += "REORIENT_TO_REVERSE_FLOW + "
                    current_flow = -1
                    
                # Fetch the original action it took to go from nxt -> curr
                original_action = self.graph_data[nxt].get('action', ['IDLE'])[0]
                
                # Invert the original movement commands to look in the direction of travel
                if original_action == 'FORWARD':
                    step_intent += 'FORWARD'
                elif original_action == 'LEFT':
                    step_intent += 'RIGHT'
                elif original_action == 'RIGHT':
                    step_intent += 'LEFT'
                else:
                    step_intent += original_action
                    
                strategy.append(step_intent)
            
            # 3. Visual Jump / Shortcut
            else:
                if i + 2 < len(path):
                    next_nxt = path[i+2]
                    if next_nxt == nxt + 1:
                        strategy.append(f"JUMP_AND_ALIGN_FORWARD_FLOW (Jump {curr}->{nxt})")
                        current_flow = 1
                    elif next_nxt == nxt - 1:
                        strategy.append(f"JUMP_AND_ALIGN_REVERSE_FLOW (Jump {curr}->{nxt})")
                        current_flow = -1
                    else:
                        strategy.append(f"JUMP_AND_ALIGN_UNKNOWN (Jump {curr}->{nxt})")
                else:
                    strategy.append(f"JUMP_{curr}_TO_{nxt}")

        return strategy
    
    def _optimize_kinematics(self, raw_intents):
        """
        Intercepts sustained backward travel and rewrites the sequence 
        to execute a 180-degree turn followed by forward-facing travel.
        """
        optimized = []
        is_reversed = False

        for action in raw_intents:
            # Pass through complex realignments and reset our orientation state.
            # The local controller will handle visual vanishing points here.
            if "JUMP" in action or "REORIENT" in action:
                optimized.append(action)
                is_reversed = False 
                continue

            if action == 'BACKWARD':
                if not is_reversed:
                    # First time we see backward, inject the 180 turn
                    optimized.append('TURN_LEFT_180')
                    is_reversed = True
                else:
                    # Already turned around; backward travel on the map is now forward travel for the chassis
                    optimized.append('FORWARD')
            else:
                if is_reversed:
                    # If we are physically turned around, our relative left/right map is flipped
                    if action == 'LEFT': 
                        optimized.append('RIGHT')
                    elif action == 'RIGHT': 
                        optimized.append('LEFT')
                    elif action == 'FORWARD': 
                        # The map says go forward, but we are facing backward. Turn around again.
                        optimized.append('TURN_LEFT_180')
                        is_reversed = False
                else:
                    optimized.append(action)
                    
        return optimized

    def distill_directional_intent(self, intents):
        """
        Parses the upcoming lookahead sequence to provide a single, actionable 
        directional intent vector for the local controller's FSM.
        """
        if not intents:
            return "STAY"
            
        immediate_action = intents[0]
        
        # 1. If the immediate action is a complex maneuver, it becomes the sole focus
        if "JUMP" in immediate_action or "REORIENT" in immediate_action:
            return immediate_action
            
        # 2. Count how many consecutive nodes share this exact same physical action
        consecutive_count = 1
        for action in intents[1:]:
            if action == immediate_action:
                consecutive_count += 1
            else:
                break
                
        # 3. Look ahead to find the NEXT major state change 
        upcoming_maneuver = None
        steps_until = None
        for i, action in enumerate(intents[consecutive_count:], start=consecutive_count):
            if action != immediate_action and action != "IDLE":
                upcoming_maneuver = action
                steps_until = i
                break
                
        # 4. Construct the final intent vector decision
        intent_vector = immediate_action
        
        # Add spatial awareness to help the local controller prepare its state machine
        if upcoming_maneuver:
            # Clean up the string slightly if it's a compound reorient command
            clean_maneuver = upcoming_maneuver.split(' + ')[-1] if ' + ' in upcoming_maneuver else upcoming_maneuver
            intent_vector += f" | NEXT: {clean_maneuver} (in {steps_until} nodes)"
            
        return intent_vector