import numpy as np
from collections import deque
from game.enums import Direction, loc_after_direction

class GraphManager:
    """
    The Strategist: Analyzes the board topology to identify territory control,
    bottlenecks, and cut-vertices.
    """

    def __init__(self, map_size=8):
        self.map_size = map_size
        self.directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    def analyze(self, board, my_color_is_white):
        """
        Performs a full topological analysis of the current board state.
        
        Returns a dictionary containing:
        - 'my_reachable': Set of squares I can reach.
        - 'opp_reachable': Set of squares opponent can reach.
        - 'my_territory': Count of squares closer to me.
        - 'opp_territory': Count of squares closer to opponent.
        - 'articulation_points': List of squares that are critical bottlenecks for the opponent.
        """
        
        # 1. Identify raw blockages for both players
        # We perform this manually to be faster than calling board.is_valid_move() 64 times
        my_blocked = self._get_blocked_mask(board, is_enemy=False)
        opp_blocked = self._get_blocked_mask(board, is_enemy=True)

        # 2. Get current locations
        my_loc = board.chicken_player.get_location()
        opp_loc = board.chicken_enemy.get_location()

        # 3. Reachability Analysis (Flood Fill)
        my_reachable_set = self._bfs_reachable(my_loc, my_blocked)
        opp_reachable_set = self._bfs_reachable(opp_loc, opp_blocked)

        # 4. Voronoi Partitioning (Territory Control)
        my_territory, opp_territory = self._compute_voronoi(my_loc, opp_loc, my_blocked, opp_blocked)

        # 5. Articulation Point Detection (The "Anaconda" Logic)
        # We want to find points in the OPPONENT'S graph that are critical.
        cut_vertices = self._find_articulation_points(opp_loc, opp_reachable_set, opp_blocked)

        return {
            'my_reachable': my_reachable_set,
            'opp_reachable': opp_reachable_set,
            'my_territory': my_territory,
            'opp_territory': opp_territory,
            'articulation_points': cut_vertices,
            'my_mobility': len(my_reachable_set),
            'opp_mobility': len(opp_reachable_set)
        }

    def simulate_turd_impact(self, board, turd_loc):
        """
        Calculates the strategic value of placing a turd at 'turd_loc'.
        Returns the number of squares removed from the opponent's reachable set.
        Higher number = Better Turd.
        """
        # 1. Current connectivity
        opp_loc = board.chicken_enemy.get_location()
        current_blocked = self._get_blocked_mask(board, is_enemy=True)
        current_reachable = self._bfs_reachable(opp_loc, current_blocked)
        
        # 2. Simulated connectivity
        # Placing a turd at turd_loc blocks turd_loc AND all its neighbors for the opponent
        simulated_blocked = current_blocked.copy()
        
        # Block the turd center
        r, c = turd_loc
        simulated_blocked[r, c] = True
        
        # Block neighbors (The "Radioactive" Zone)
        for d in self.directions:
            nr, nc = loc_after_direction(turd_loc, d)
            if 0 <= nr < self.map_size and 0 <= nc < self.map_size:
                simulated_blocked[nr, nc] = True
                
        new_reachable = self._bfs_reachable(opp_loc, simulated_blocked)
        
        impact = len(current_reachable) - len(new_reachable)
        return impact

    def _get_blocked_mask(self, board, is_enemy):
        """
        Constructs a boolean grid where True = Blocked/Impassable.
        This manually implements the game rules for movement validity to be blazing fast.
        """
        mask = np.zeros((self.map_size, self.map_size), dtype=bool)
        
        # Get relevant sets based on perspective
        if is_enemy:
            # Enemy cannot step on:
            # 1. My Eggs
            # 2. My Turds
            # 3. Adjacent to My Turds
            # 4. The square I occupy (collision)
            eggs = board.eggs_player
            turds = board.turds_player
            blocker_loc = board.chicken_player.get_location()
        else:
            # I cannot step on:
            # 1. Enemy Eggs
            # 2. Enemy Turds
            # 3. Adjacent to Enemy Turds
            # 4. The square Enemy occupies
            eggs = board.eggs_enemy
            turds = board.turds_enemy
            blocker_loc = board.chicken_enemy.get_location()

        # Mark simple obstructions
        for (r, c) in eggs:
            mask[r, c] = True
            
        mask[blocker_loc[0], blocker_loc[1]] = True

        # Mark Turd Zones (Center + Neighbors)
        # Rule: "may not move into any square that shares an edge with... opponent's turds"
        for (tr, tc) in turds:
            mask[tr, tc] = True # The turd itself
            for d in self.directions:
                nr, nc = loc_after_direction((tr, tc), d)
                if 0 <= nr < self.map_size and 0 <= nc < self.map_size:
                    mask[nr, nc] = True

        return mask

    def _bfs_reachable(self, start_loc, blocked_mask):
        """
        Standard BFS to find all reachable nodes in the graph component.
        """
        if blocked_mask[start_loc[0], start_loc[1]]:
            return set()

        q = deque([start_loc])
        visited = {start_loc}
        
        while q:
            curr = q.popleft()
            for d in self.directions:
                next_loc = loc_after_direction(curr, d)
                nx, ny = next_loc
                
                # Bounds Check
                if not (0 <= nx < self.map_size and 0 <= ny < self.map_size):
                    continue
                
                # Blockage Check
                if blocked_mask[nx, ny]:
                    continue
                    
                if next_loc not in visited:
                    visited.add(next_loc)
                    q.append(next_loc)
        
        return visited

    def _compute_voronoi(self, my_loc, opp_loc, my_blocked, opp_blocked):
        """
        Simultaneous BFS to determine territory control.
        """
        # Queue items: (r, c, distance, owner_id)
        # owner_id: 0 = Me, 1 = Opponent
        q = deque()
        
        # Territory grids (stores min_dist to reach)
        # Initialize with infinity
        my_dists = np.full((self.map_size, self.map_size), 999)
        opp_dists = np.full((self.map_size, self.map_size), 999)
        
        if not my_blocked[my_loc[0], my_loc[1]]:
            q.append((my_loc, 0, 0))
            my_dists[my_loc] = 0
            
        if not opp_blocked[opp_loc[0], opp_loc[1]]:
            q.append((opp_loc, 0, 1))
            opp_dists[opp_loc] = 0
            
        while q:
            curr_loc, dist, owner = q.popleft()
            
            # Select the correct blockage map and distance grid for the current owner
            if owner == 0:
                block_map = my_blocked
                dist_map = my_dists
            else:
                block_map = opp_blocked
                dist_map = opp_dists
                
            new_dist = dist + 1
            
            for d in self.directions:
                next_loc = loc_after_direction(curr_loc, d)
                nx, ny = next_loc
                
                if not (0 <= nx < self.map_size and 0 <= ny < self.map_size):
                    continue
                
                if block_map[nx, ny]:
                    continue
                
                # If we found a shorter path to this square, update and continue
                if new_dist < dist_map[nx, ny]:
                    dist_map[nx, ny] = new_dist
                    q.append((next_loc, new_dist, owner))

        # Count territory
        # A square is "mine" if my_dist < opp_dist.
        # Note: We only care about squares that represent valid territory (not blocked).
        # However, purely geometric Voronoi is often good enough for heuristics.
        
        my_count = 0
        opp_count = 0
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                d_my = my_dists[r, c]
                d_opp = opp_dists[r, c]
                
                if d_my == 999 and d_opp == 999:
                    continue # Unreachable by both
                
                if d_my < d_opp:
                    my_count += 1
                elif d_opp < d_my:
                    opp_count += 1
                # Ties ignored (contested)
                
        return my_count, opp_count

    def _find_articulation_points(self, start_loc, reachable_set, blocked_mask):
        """
        Uses Tarjan's algorithm (or simple recursive DFS) to find cut vertices
        in the reachable subgraph.
        """
        if not reachable_set:
            return []

        # Convert set to list for indexing
        nodes = list(reachable_set)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        
        adj = [[] for _ in range(n)]
        
        # Build adjacency list for the subgraph
        for u_loc in nodes:
            u_idx = node_to_idx[u_loc]
            for d in self.directions:
                v_loc = loc_after_direction(u_loc, d)
                if v_loc in reachable_set:
                    # It's a valid edge in the subgraph
                    v_idx = node_to_idx[v_loc]
                    adj[u_idx].append(v_idx)
                    
        # Tarjan's Algo variables
        visited = [False] * n
        disc = [-1] * n
        low = [-1] * n
        parent = [-1] * n
        ap = [False] * n
        self.time = 0

        def APUtil(u):
            children = 0
            visited[u] = True
            disc[u] = self.time
            low[u] = self.time
            self.time += 1

            for v in adj[u]:
                if not visited[v]:
                    children += 1
                    parent[v] = u
                    APUtil(v)
                    low[u] = min(low[u], low[v])

                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        # Run DFS from the start location (root of the component)
        if start_loc in node_to_idx:
            root_idx = node_to_idx[start_loc]
            APUtil(root_idx)
            if (list(adj[root_idx]).__len__() > 1): # Correct check for root in Tarjan's
                 # Actually, for root, if children > 1 it is AP.
                 # The recursive function handles non-root.
                 pass
            # Re-check root case manually:
            # If root has > 1 child in the DFS tree, it's an AP.
            # My APUtil logic for root is slightly implicit, let's trust the standard non-root check 
            # and handle root manually if needed. 
            # In this game context, "root" is the opponent's current location. 
            # If the opponent is standing ON a cut vertex, they are already in trouble, 
            # but we can't "cut" the node they are standing on. 
            # We want to cut OTHER nodes.
            ap[root_idx] = False # You cannot cut the vertex the agent is currently standing on to split them.

        articulation_points = []
        for i in range(n):
            if ap[i]:
                articulation_points.append(nodes[i])
                
        return articulation_points