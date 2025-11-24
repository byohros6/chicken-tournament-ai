"""
A* Pathfinding for Egg Tour Optimization
Computes optimal paths considering trapdoor risk and obstacles
"""
import heapq
import math
from typing import List, Tuple, Set, Optional, Dict
from game.board import Board
from game.enums import Direction, loc_after_direction


class PathFinder:
    """
    A* pathfinding with trapdoor risk and obstacle avoidance.
    """
    
    def __init__(self, board: Board):
        self.map_size = board.game_map.MAP_SIZE
    
    def manhattan_distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    
    def euclidean_distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic (admissible)."""
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def astar_path(self, board: Board, start: Tuple[int, int], goal: Tuple[int, int],
                   trapdoor_risks: Dict[Tuple[int, int], float]) -> Optional[List[Direction]]:
        """
        A* search from start to goal, avoiding obstacles and high-risk trapdoors.
        
        Returns:
            List of Directions to reach goal, or None if no path exists
        """
        if start == goal:
            return []
        
        # Priority queue: (f_score, g_score, location)
        open_set = [(0.0, 0.0, start)]
        
        # Track best g_score to each location
        g_scores = {start: 0.0}
        
        # Track path: location -> (previous_location, direction_taken)
        came_from: Dict[Tuple[int, int], Tuple[Tuple[int, int], Direction]] = {}
        
        while open_set:
            f_score, g_score, current = heapq.heappop(open_set)
            
            # Reached goal?
            if current == goal:
                return self._reconstruct_path(came_from, start, goal)
            
            # Check if this is outdated (we found a better path already)
            if current in g_scores and g_score > g_scores[current]:
                continue
            
            # Explore neighbors
            for direction in Direction:
                neighbor = loc_after_direction(current, direction)
                
                # Valid cell?
                if not board.is_valid_cell(neighbor):
                    continue
                
                # Blocked?
                if board.is_cell_blocked(neighbor):
                    continue
                
                # Calculate cost to move to neighbor
                move_cost = 1.0
                
                # Add trapdoor risk penalty (makes high-risk squares "expensive")
                trap_risk = trapdoor_risks.get(neighbor, 0.0)
                move_cost += trap_risk * 500.0  # VERY high penalty - avoid risky squares at all costs
                
                # Calculate tentative g_score
                tentative_g = g_score + move_cost
                
                # Is this a better path to neighbor?
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    came_from[neighbor] = (current, direction)
                    
                    # f_score = g_score + heuristic
                    h_score = self.manhattan_distance(neighbor, goal)
                    f_score_new = tentative_g + h_score
                    
                    heapq.heappush(open_set, (f_score_new, tentative_g, neighbor))
        
        # No path found
        return None
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[Tuple[int, int], Direction]],
                          start: Tuple[int, int], goal: Tuple[int, int]) -> List[Direction]:
        """Reconstruct path from came_from map."""
        path = []
        current = goal
        
        while current != start:
            if current not in came_from:
                return []  # Path broken
            prev, direction = came_from[current]
            path.append(direction)
            current = prev
        
        path.reverse()
        return path
    
    def compute_egg_tour(self, board: Board, my_parity: int,
                        trapdoor_risks: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
        """
        Compute an optimized tour through all egg-laying squares.
        Uses greedy nearest-neighbor with A* paths.
        
        Returns:
            Ordered list of locations to visit for egg laying
        """
        # Get all valid egg-laying squares for this parity
        egg_squares = []
        for x in range(self.map_size):
            for y in range(self.map_size):
                loc = (x, y)
                if (x + y) % 2 == my_parity:
                    # Check if accessible (not permanently blocked)
                    if loc not in board.eggs_enemy and loc not in board.turds_enemy:
                        egg_squares.append(loc)
        
        if not egg_squares:
            return []
        
        # Greedy tour construction: always go to nearest unvisited square
        current = board.chicken_player.get_location()
        tour = []
        unvisited = set(egg_squares)
        
        # Prioritize corners first (they give 3 eggs each)
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        corner_parity_squares = [c for c in corners if c in unvisited]
        
        # Visit corners first if they're accessible
        for corner in corner_parity_squares:
            # Check if we can reach it
            path = self.astar_path(board, current, corner, trapdoor_risks)
            if path is not None:
                tour.append(corner)
                unvisited.remove(corner)
                current = corner
        
        # Then visit remaining squares in nearest-neighbor order
        while unvisited:
            # Find nearest unvisited square (by A* path cost, not just manhattan)
            best_next = None
            best_path_cost = float('inf')
            
            for candidate in unvisited:
                # Estimate cost (use manhattan + trapdoor risk as quick heuristic)
                dist = self.manhattan_distance(current, candidate)
                trap_risk = trapdoor_risks.get(candidate, 0.0)
                estimated_cost = dist + trap_risk * 50.0
                
                if estimated_cost < best_path_cost:
                    best_path_cost = estimated_cost
                    best_next = candidate
            
            if best_next is None:
                break  # No reachable squares
            
            tour.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        
        return tour
    
    def optimize_tour_2opt(self, tour: List[Tuple[int, int]], 
                           board: Board,
                           trapdoor_risks: Dict[Tuple[int, int], float],
                           max_iterations: int = 50) -> List[Tuple[int, int]]:
        """
        Improve tour using 2-opt swaps.
        Limited iterations to avoid timeout.
        """
        if len(tour) < 4:
            return tour
        
        improved = True
        iteration = 0
        current_tour = tour[:]
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(current_tour) - 2):
                for j in range(i + 2, len(current_tour)):
                    # Try swapping edge (i, i+1) and (j, j+1)
                    # Original: ... -> tour[i] -> tour[i+1] -> ... -> tour[j] -> tour[j+1] -> ...
                    # Swapped:  ... -> tour[i] -> tour[j] -> ... -> tour[i+1] -> tour[j+1] -> ...
                    
                    # Calculate cost before swap
                    cost_before = self._tour_segment_cost(current_tour, i, j, trapdoor_risks)
                    
                    # Create swapped tour
                    new_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
                    
                    # Calculate cost after swap
                    cost_after = self._tour_segment_cost(new_tour, i, j, trapdoor_risks)
                    
                    if cost_after < cost_before:
                        current_tour = new_tour
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_tour
    
    def _tour_segment_cost(self, tour: List[Tuple[int, int]], i: int, j: int,
                          trapdoor_risks: Dict[Tuple[int, int], float]) -> float:
        """Calculate cost of tour segment for 2-opt comparison."""
        cost = 0.0
        
        # Cost from tour[i] to tour[i+1]
        if i + 1 < len(tour):
            dist = self.manhattan_distance(tour[i], tour[i+1])
            risk = trapdoor_risks.get(tour[i+1], 0.0)
            cost += dist + risk * 50.0
        
        # Cost from tour[j] to tour[j+1]
        if j + 1 < len(tour):
            dist = self.manhattan_distance(tour[j], tour[j+1])
            risk = trapdoor_risks.get(tour[j+1], 0.0)
            cost += dist + risk * 50.0
        
        return cost
