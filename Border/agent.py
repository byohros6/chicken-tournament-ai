from collections.abc import Callable
from typing import List, Set, Tuple
import numpy as np
from game import *
from game.enums import Direction, MoveType, loc_after_direction


class TrapdoorBelief:
    """
    Maintains probability distributions over trapdoor locations using Bayesian inference.
    """
    
    def __init__(self, map_size: int = 8):
        self.map_size = map_size
        # Probability grids for even and odd trapdoors
        # Initialize with uniform distribution over valid squares (weighted by distance from edge)
        self.even_probs = np.ones((map_size, map_size)) * 0.0
        self.odd_probs = np.ones((map_size, map_size)) * 0.0
        
        # Initialize weights based on distance from edge (as per trapdoor_manager.py)
        weights = np.zeros((map_size, map_size))
        weights[2:map_size-2, 2:map_size-2] = 1.0
        weights[3:map_size-3, 3:map_size-3] = 2.0
        
        # Normalize for even squares (i+j is even)
        even_weights = weights.copy()
        for i in range(map_size):
            for j in range(map_size):
                if (i + j) % 2 != 0:
                    even_weights[i, j] = 0.0
        
        # Normalize for odd squares (i+j is odd)
        odd_weights = weights.copy()
        for i in range(map_size):
            for j in range(map_size):
                if (i + j) % 2 != 1:
                    odd_weights[i, j] = 0.0
        
        # Normalize to probabilities
        even_sum = np.sum(even_weights)
        odd_sum = np.sum(odd_weights)
        if even_sum > 0:
            self.even_probs = even_weights / even_sum
        if odd_sum > 0:
            self.odd_probs = odd_weights / odd_sum
    
    def update(self, location: Tuple[int, int], sensor_data: List[Tuple[bool, bool]], 
               chicken: 'Chicken'):
        """
        Update trapdoor probabilities using Bayesian inference.
        
        Args:
            location: Current location (x, y)
            sensor_data: [(heard_even, felt_even), (heard_odd, felt_odd)]
            chicken: Chicken object for probability calculations
        """
        # Update even trapdoor probabilities
        heard_even, felt_even = sensor_data[0]
        likelihood_even = np.ones((self.map_size, self.map_size))
        
        for i in range(self.map_size):
            for j in range(self.map_size):
                if (i + j) % 2 == 0:  # Even square
                    hear_prob, feel_prob = chicken.prob_senses_if_trapdoor_were_at(
                        heard_even, felt_even, i, j
                    )
                    likelihood_even[i, j] = hear_prob * feel_prob
                else:
                    likelihood_even[i, j] = 0.0
        
        # Bayesian update for even trapdoor
        self.even_probs = self.even_probs * likelihood_even
        even_sum = np.sum(self.even_probs)
        if even_sum > 0:
            self.even_probs = self.even_probs / even_sum
        
        # Update odd trapdoor probabilities
        heard_odd, felt_odd = sensor_data[1]
        likelihood_odd = np.ones((self.map_size, self.map_size))
        
        for i in range(self.map_size):
            for j in range(self.map_size):
                if (i + j) % 2 == 1:  # Odd square
                    hear_prob, feel_prob = chicken.prob_senses_if_trapdoor_were_at(
                        heard_odd, felt_odd, i, j
                    )
                    likelihood_odd[i, j] = hear_prob * feel_prob
                else:
                    likelihood_odd[i, j] = 0.0
        
        # Bayesian update for odd trapdoor
        self.odd_probs = self.odd_probs * likelihood_odd
        odd_sum = np.sum(self.odd_probs)
        if odd_sum > 0:
            self.odd_probs = self.odd_probs / odd_sum
    
    def get_trapdoor_risk(self, location: Tuple[int, int]) -> float:
        """
        Get the combined probability that a trapdoor is at this location.
        """
        x, y = location
        if (x + y) % 2 == 0:
            return self.even_probs[x, y]
        else:
            return self.odd_probs[x, y]
    
    def get_max_risk_locations(self, threshold: float = 0.1) -> Set[Tuple[int, int]]:
        """
        Get set of locations with trapdoor risk above threshold.
        """
        risky = set()
        for i in range(self.map_size):
            for j in range(self.map_size):
                risk = self.get_trapdoor_risk((i, j))
                if risk > threshold:
                    risky.add((i, j))
        return risky


class PlayerAgent:
    """
    A strategic chicken agent that uses trapdoor inference, move evaluation, and search.
    """
    
    def __init__(self, board: board.Board, time_left: Callable):
        """
        Initialize the agent with trapdoor belief system.
        """
        self.map_size = board.game_map.MAP_SIZE
        self.trapdoor_belief = TrapdoorBelief(self.map_size)
        self.starting_location = board.chicken_player.get_location()
        self.even_chicken = board.chicken_player.even_chicken
        
        # Determine if we are Player A or Player B
        # Player A: even_chicken == 0, goes first, can lay eggs on even squares (0,0), (0,2), etc.
        # Player B: even_chicken == 1, goes second, can lay eggs on odd squares (0,1), (1,0), etc.
        self.is_player_a = board.chicken_player.is_player_a()
        self.player_label = "A" if self.is_player_a else "B"
        
        self.time_budget_per_move = 8.0  # ~9 seconds per move on average
        self.moves_made = 0
        
        # Track visited locations (to avoid revisiting squares)
        self.visited_locations = set()
        self.visited_locations.add(self.starting_location)
        
        # Create grid map of all egg-laying squares we can visit
        self.egg_laying_squares = set()
        for x in range(self.map_size):
            for y in range(self.map_size):
                if self.can_lay_egg_here((x, y)):
                    self.egg_laying_squares.add((x, y))
        
        # Track which egg-laying squares we've already laid eggs on
        self.eggs_laid_squares = set()
        
        # Debug: Track egg laying
        self.eggs_laid_log = []  # List of (location, is_corner, turn_number, eggs_before, eggs_after)
        self.last_egg_count = 0
        
        print(f"[MyAgent | INIT]: I am Player {self.player_label}, starting at {self.starting_location}, "
              f"even_chicken={self.even_chicken}, total egg squares: {len(self.egg_laying_squares)}")
    
    def is_corner(self, location: Tuple[int, int]) -> bool:
        """Check if location is a corner."""
        x, y = location
        return (x == 0 or x == self.map_size - 1) and (y == 0 or y == self.map_size - 1)
    
    def can_lay_egg_here(self, location: Tuple[int, int]) -> bool:
        """Check if we can lay an egg at this location based on our parity."""
        x, y = location
        if self.is_player_a:
            # Player A: can lay on even squares (x + y is even)
            return (x + y) % 2 == 0
        else:
            # Player B: can lay on odd squares (x + y is odd)
            return (x + y) % 2 == 1
    
    def manhattan_distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two locations."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    
    def distance_to_center(self, location: Tuple[int, int]) -> float:
        """Calculate distance to center of board. Lower is better."""
        center = (self.map_size / 2 - 0.5, self.map_size / 2 - 0.5)
        x, y = location
        return abs(x - center[0]) + abs(y - center[1])
    
    def find_nearest_unvisited_egg_square(self, current_loc: Tuple[int, int], board: board.Board) -> Tuple[int, int]:
        """
        Find the nearest unvisited egg-laying square using BFS.
        Returns the target square or None if all are visited.
        """
        # Get unvisited egg-laying squares (haven't laid eggs there yet)
        unvisited_egg_squares = []
        for square in self.egg_laying_squares:
            # Check if we haven't laid an egg there and it's not blocked
            if square not in self.eggs_laid_squares and square not in board.eggs_enemy:
                # Check if it's accessible (not blocked by enemy turds)
                if not board.is_cell_blocked(square):
                    unvisited_egg_squares.append(square)
        
        if not unvisited_egg_squares:
            return None
        
        # Use BFS to find nearest unvisited egg square
        from collections import deque
        queue = deque([(current_loc, 0)])
        visited_bfs = {current_loc}
        
        while queue:
            loc, dist = queue.popleft()
            
            # Check if this is an unvisited egg square
            if loc in unvisited_egg_squares:
                return loc
            
            # Explore neighbors
            for direction in Direction:
                new_loc = loc_after_direction(loc, direction)
                if (board.is_valid_cell(new_loc) and 
                    new_loc not in visited_bfs and
                    not board.is_cell_blocked(new_loc)):
                    visited_bfs.add(new_loc)
                    queue.append((new_loc, dist + 1))
        
        return None
    
    def find_path_to_target(self, current_loc: Tuple[int, int], target_loc: Tuple[int, int], 
                           board: board.Board) -> Direction:
        """
        Find the best direction to move toward target using BFS.
        Returns the direction to move, or None if no path exists.
        """
        if current_loc == target_loc:
            return None
        
        from collections import deque
        queue = deque([(current_loc, [])])
        visited_bfs = {current_loc}
        
        while queue:
            loc, path = queue.popleft()
            
            if loc == target_loc:
                # Return first direction in path
                if path:
                    return path[0]
                return None
            
            # Explore neighbors
            for direction in Direction:
                new_loc = loc_after_direction(loc, direction)
                if (board.is_valid_cell(new_loc) and 
                    new_loc not in visited_bfs and
                    not board.is_cell_blocked(new_loc)):
                    visited_bfs.add(new_loc)
                    queue.append((new_loc, path + [direction]))
        
        return None
    
    def evaluate_move(self, board: board.Board, direction: Direction, 
                     move_type: MoveType, trapdoor_belief: TrapdoorBelief) -> float:
        """
        Evaluate a move and return a score. Higher is better.
        Simple strategy: Prioritize eggs, avoid visited squares, move toward center.
        """
        current_loc = board.chicken_player.get_location()
        new_loc = loc_after_direction(current_loc, direction)
        
        if not board.is_valid_cell(new_loc):
            return float('-inf')
        
        score = 0.0
        
        # Trapdoor risk penalty (very high penalty)
        trapdoor_risk = trapdoor_belief.get_trapdoor_risk(new_loc)
        score -= trapdoor_risk * 1000.0
        
        # PRIORITY 1: Lay eggs whenever possible
        if move_type == MoveType.EGG:
            if self.can_lay_egg_here(current_loc) and board.can_lay_egg_at_loc(current_loc):
                score += 1000.0  # Very high priority
                # Corner eggs are worth 3x
                if self.is_corner(current_loc):
                    score += 500.0  # Extra bonus for corner eggs
        
        # PRIORITY 2: Avoid going back to visited squares
        if move_type == MoveType.PLAIN:
            if new_loc in self.visited_locations:
                score -= 2000.0  # Very strong penalty for revisiting (should almost never happen)
        
        # PRIORITY 3: Move toward center (not edges), but prioritize corners
        if move_type == MoveType.PLAIN:
            # Check if new_loc is a corner we can lay eggs on
            if self.is_corner(new_loc) and self.can_lay_egg_here(new_loc):
                if new_loc not in board.eggs_player:  # Haven't laid egg there yet
                    score += 100.0  # High priority for corner squares
            
            # Prefer moving toward center
            current_dist_to_center = self.distance_to_center(current_loc)
            new_dist_to_center = self.distance_to_center(new_loc)
            if new_dist_to_center < current_dist_to_center:
                score += 10.0  # Bonus for moving toward center
            elif new_dist_to_center > current_dist_to_center:
                score -= 5.0  # Penalty for moving away from center
            
            # Prefer moving to squares where we can lay eggs
            if self.can_lay_egg_here(new_loc):
                score += 20.0
        
        # Prefer moves that don't put us near enemy turds
        if board.is_cell_in_enemy_turd_zone(new_loc):
            score -= 50.0
        
        return score
    
    def get_corner_squares(self) -> List[Tuple[int, int]]:
        """Get all corner squares."""
        return [
            (0, 0), (0, self.map_size - 1),
            (self.map_size - 1, 0), (self.map_size - 1, self.map_size - 1)
        ]
    
    def get_best_move_greedy(self, board: board.Board, trapdoor_belief: TrapdoorBelief) -> Tuple[Direction, MoveType]:
        """
        Simple greedy strategy:
        1. Prioritize laying eggs (especially corner eggs)
        2. Move to nearby corner squares if we can lay eggs there
        3. Avoid visited squares
        4. Move toward center
        """
        current_loc = board.chicken_player.get_location()
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Check enemy distance
        enemy_loc = board.chicken_enemy.get_location()
        enemy_distance = self.manhattan_distance(current_loc, enemy_loc)
        
        # PRIORITY 0: If enemy is within 2 blocks (Manhattan distance <= 2), lay a turd to block them
        # Turd blocks 4 squares (turd square + 3 adjacent) where enemy could lay eggs
        # IMPORTANT: Only lay turds when enemy is actually close!
        if enemy_distance <= 2 and enemy_distance > 1:  # Must be > 1 because can't lay turd adjacent to enemy
            can_lay_turd_here = board.can_lay_turd_at_loc(current_loc) and board.chicken_player.has_turds_left()
            if can_lay_turd_here:
                turd_moves = [(d, mt) for d, mt in valid_moves if mt == MoveType.TURD]
                if turd_moves:
                    # Pick the best direction to move after laying the turd
                    best_turd = None
                    best_turd_score = float('-inf')
                    for direction, move_type in turd_moves:
                        score = self.evaluate_move(board, direction, move_type, trapdoor_belief)
                        if score > best_turd_score:
                            best_turd_score = score
                            best_turd = (direction, move_type)
                    if best_turd:
                        return best_turd
        
        # PRIORITY 1: If we're at a valid egg-laying square, lay an egg
        # Egg Step: lays egg at CURRENT location, then moves in chosen direction
        can_lay_egg_here = self.can_lay_egg_here(current_loc) and board.can_lay_egg_at_loc(current_loc)
        if can_lay_egg_here:
            egg_moves = [(d, mt) for d, mt in valid_moves if mt == MoveType.EGG]
            if egg_moves:
                # Pick the best direction to move after laying the egg
                best_egg = None
                best_egg_score = float('-inf')
                for direction, move_type in egg_moves:
                    score = self.evaluate_move(board, direction, move_type, trapdoor_belief)
                    if score > best_egg_score:
                        best_egg_score = score
                        best_egg = (direction, move_type)
                if best_egg:
                    return best_egg
        
        # PRIORITY 2: Check if there's a corner square within 2 blocks we can lay an egg on
        # Move toward it if we're not already at a valid egg-laying square
        if not can_lay_egg_here:
            corner_squares = self.get_corner_squares()
            for corner in corner_squares:
                # Check if it's a valid egg-laying square for us
                if self.can_lay_egg_here(corner):
                    # Check if we haven't already laid an egg there
                    if corner not in board.eggs_player:
                        # Check distance
                        dist_to_corner = self.manhattan_distance(current_loc, corner)
                        if dist_to_corner <= 2:
                            # Try to find a move that gets us closer to this corner
                            best_move_to_corner = None
                            best_corner_score = float('-inf')
                            
                            for direction, move_type in valid_moves:
                                if move_type == MoveType.PLAIN:
                                    new_loc = loc_after_direction(current_loc, direction)
                                    if new_loc not in self.visited_locations:
                                        # Calculate how much closer we get
                                        current_dist = self.manhattan_distance(current_loc, corner)
                                        new_dist = self.manhattan_distance(new_loc, corner)
                                        
                                        if new_dist < current_dist:
                                            score = 2000.0 - new_dist * 100  # Higher score for closer
                                            # Extra bonus if we can reach it in one move
                                            if new_dist == 0:
                                                score += 1000.0  # We're at the corner!
                                            
                                            if score > best_corner_score:
                                                best_corner_score = score
                                                best_move_to_corner = (direction, move_type)
                            
                            if best_move_to_corner:
                                return best_move_to_corner
        
        # PRIORITY 3: Use pathfinding to move toward nearest unvisited egg-laying square
        # This prevents oscillation by having a clear target
        target_square = self.find_nearest_unvisited_egg_square(current_loc, board)
        if target_square:
            # Find path to target
            target_direction = self.find_path_to_target(current_loc, target_square, board)
            if target_direction:
                # Check if this direction is valid
                for direction, move_type in valid_moves:
                    if direction == target_direction and move_type == MoveType.PLAIN:
                        new_loc = loc_after_direction(current_loc, direction)
                        # Prefer unvisited squares
                        if new_loc not in self.visited_locations:
                            return (direction, move_type)
                        # If visited, still use it but with lower priority
                        return (direction, move_type)
        
        # FALLBACK: Move to unvisited squares WITHOUT eggs, preferably toward center
        unvisited_moves_no_eggs = []
        unvisited_moves_with_eggs = []
        visited_moves = []
        
        for direction, move_type in valid_moves:
            if move_type == MoveType.PLAIN:
                new_loc = loc_after_direction(current_loc, direction)
                # Check if we've visited this square
                if new_loc in self.visited_locations:
                    visited_moves.append((direction, move_type))
                else:
                    # Check if there's already an egg there
                    if new_loc not in board.eggs_player:
                        unvisited_moves_no_eggs.append((direction, move_type))
                    else:
                        unvisited_moves_with_eggs.append((direction, move_type))
            # Skip turds - only lay them when enemy is close (Priority 0)
            elif move_type == MoveType.TURD:
                pass  # Don't consider turds here
        
        # Prioritize: unvisited without eggs > unvisited with eggs > visited (only if no other option)
        moves_to_evaluate = []
        if unvisited_moves_no_eggs:
            moves_to_evaluate = unvisited_moves_no_eggs
        elif unvisited_moves_with_eggs:
            moves_to_evaluate = unvisited_moves_with_eggs
        else:
            # Only use visited moves if absolutely no other option
            moves_to_evaluate = visited_moves if visited_moves else valid_moves
        
        # Evaluate all moves and pick best
        best_move = None
        best_score = float('-inf')
        
        for direction, move_type in moves_to_evaluate:
            score = self.evaluate_move(board, direction, move_type, trapdoor_belief)
            # Extra penalty for visited squares
            if move_type == MoveType.PLAIN:
                new_loc = loc_after_direction(current_loc, direction)
                if new_loc in self.visited_locations:
                    score -= 1000.0  # Very strong penalty for revisiting
            
            if score > best_score:
                best_score = score
                best_move = (direction, move_type)
        
        return best_move
    
    def get_best_move_with_search(self, board: board.Board, trapdoor_belief: TrapdoorBelief,
                                 time_left: Callable, depth: int = 2) -> Tuple[Direction, MoveType]:
        """
        Use minimax search to find best move (limited depth due to time constraints).
        """
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None
        
        # If we're running low on time, use greedy
        if time_left() < 30.0:
            return self.get_best_move_greedy(board, trapdoor_belief)
        
        best_move = None
        best_score = float('-inf')
        
        # Try each valid move
        for direction, move_type in valid_moves:
            # Forecast the move
            forecast_board = board.forecast_move(direction, move_type, check_ok=True)
            if forecast_board is None:
                continue
            
            # Evaluate the resulting position
            score = self.evaluate_position(forecast_board, trapdoor_belief, depth - 1)
            
            if score > best_score:
                best_score = score
                best_move = (direction, move_type)
        
        return best_move if best_move else self.get_best_move_greedy(board, trapdoor_belief)
    
    def evaluate_position(self, board: board.Board, trapdoor_belief: TrapdoorBelief, 
                         depth: int = 0) -> float:
        """
        Evaluate a board position. Returns a score from player's perspective.
        """
        if depth <= 0:
            # Base evaluation
            score = 0.0
            
            # Egg count difference (most important)
            my_eggs = board.chicken_player.get_eggs_laid()
            enemy_eggs = board.chicken_enemy.get_eggs_laid()
            score += (my_eggs - enemy_eggs) * 50.0
            
            # Trapdoor risk at current location
            my_loc = board.chicken_player.get_location()
            trapdoor_risk = trapdoor_belief.get_trapdoor_risk(my_loc)
            score -= trapdoor_risk * 100.0
            
            # Count potential egg-laying opportunities
            my_valid_moves = board.get_valid_moves(enemy=False)
            egg_opportunities = sum(1 for d, mt in my_valid_moves if mt == MoveType.EGG)
            score += egg_opportunities * 2.0
            
            # Count enemy's egg opportunities (negative)
            enemy_valid_moves = board.get_valid_moves(enemy=True)
            enemy_egg_opportunities = sum(1 for d, mt in enemy_valid_moves if mt == MoveType.EGG)
            score -= enemy_egg_opportunities * 1.5
            
            # Turd count (having more turds is slightly better)
            score += board.chicken_player.get_turds_left() * 0.5
            
            return score
        else:
            # Recursive evaluation with opponent moves
            # Simple approximation: assume opponent makes greedy move
            enemy_moves = board.get_valid_moves(enemy=True)
            if not enemy_moves:
                return self.evaluate_position(board, trapdoor_belief, 0)
            
            # Find opponent's best move (from their perspective)
            worst_score = float('inf')
            for direction, move_type in enemy_moves:
                forecast_board = board.forecast_move(direction, move_type, check_ok=True)
                if forecast_board is None:
                    continue
                forecast_board.reverse_perspective()
                score = self.evaluate_position(forecast_board, trapdoor_belief, depth - 1)
                worst_score = min(worst_score, score)
            
            return worst_score if worst_score != float('inf') else self.evaluate_position(board, trapdoor_belief, 0)
    
    def play(self, board: board.Board, sensor_data: List[Tuple[bool, bool]], 
             time_left: Callable) -> Tuple[Direction, MoveType]:
        """
        Main play method. Called on each turn.
        """
        current_location = board.chicken_player.get_location()
        current_egg_count = board.chicken_player.get_eggs_laid()
        
        # Update last position (for oscillation tracking)
        # On first move, last_position is starting_location, otherwise it's where we were last turn
        if self.moves_made > 0:
            # We're now at current_location, so last_position should be where we came from
            # But we don't know that, so we'll use current_location as reference
            pass  # Will update after move
        
        # Update trapdoor beliefs with sensor data
        self.trapdoor_belief.update(current_location, sensor_data, board.chicken_player)
        
        # Track visited locations
        self.visited_locations.add(current_location)
        
        # Determine search depth based on time and game phase
        time_remaining = time_left()
        turns_remaining = board.turns_left_player
        
        # Use more search early game, less later
        if turns_remaining > 30 and time_remaining > 200:
            search_depth = 2
        elif turns_remaining > 15 and time_remaining > 100:
            search_depth = 1
        else:
            search_depth = 0  # Greedy
        
        # Select move
        if search_depth > 0:
            move = self.get_best_move_with_search(board, self.trapdoor_belief, time_left, search_depth)
        else:
            move = self.get_best_move_greedy(board, self.trapdoor_belief)
        
        # Fallback: if no move found, use first valid move
        if move is None:
            valid_moves = board.get_valid_moves()
            if valid_moves:
                move = valid_moves[0]
            else:
                # This shouldn't happen, but handle it
                return (Direction.UP, MoveType.PLAIN)
        
        # Update last position to track oscillation
        direction, move_type = move
        
        # Log all moves for Player A
        if self.is_player_a:
            new_position = loc_after_direction(current_location, direction)
            enemy_loc = board.chicken_enemy.get_location()
            enemy_dist = self.manhattan_distance(current_location, enemy_loc)
            
            move_type_str = MoveType(move_type).name
            direction_str = Direction(direction).name
            
            print(f"[MyAgent | Player A | turn #{40 - turns_remaining}]: Move: {move_type_str} {direction_str} "
                  f"from {current_location} to {new_position}, enemy distance: {enemy_dist}")
        
        # Debug: Log egg laying
        if move_type == MoveType.EGG:
            is_corner = self.is_corner(current_location)
            eggs_before = current_egg_count
            # Estimate eggs after (will be +1 or +3 for corner)
            eggs_after_estimate = eggs_before + (3 if is_corner else 1)
            
            # Track that we've laid an egg at this square
            self.eggs_laid_squares.add(current_location)
            
            print(f"[MyAgent | Player {self.player_label} | turn #{40 - turns_remaining}]: Laying egg at {current_location}, "
                  f"corner={is_corner}, eggs: {eggs_before} -> {eggs_after_estimate}, "
                  f"remaining egg squares: {len(self.egg_laying_squares) - len(self.eggs_laid_squares)}")
            self.eggs_laid_log.append((current_location, is_corner, 40 - turns_remaining, eggs_before, eggs_after_estimate))
        
        # Debug: Log turd laying
        if move_type == MoveType.TURD:
            enemy_loc = board.chicken_enemy.get_location()
            enemy_dist = self.manhattan_distance(current_location, enemy_loc)
            print(f"[MyAgent | Player {self.player_label} | turn #{40 - turns_remaining}]: Laying turd at {current_location}, "
                  f"enemy distance: {enemy_dist}, turds left: {board.chicken_player.get_turds_left()}")
        
        # Update visited locations (for avoiding revisiting squares)
        new_position = loc_after_direction(current_location, direction)
        self.visited_locations.add(new_position)
        
        # Also update eggs_laid_squares from board state (in case we missed any)
        self.eggs_laid_squares.update(board.eggs_player)
        
        # Debug: Log current state periodically
        if self.moves_made % 10 == 0:
            eggs_on_board = len(board.eggs_player)
            print(f"[MyAgent | Player {self.player_label} | turn #{40 - turns_remaining}]: Total eggs laid: {current_egg_count}, "
                  f"Eggs on board: {eggs_on_board}, Eggs in set: {len(board.eggs_player)}")
            print(f"[MyAgent | Player {self.player_label} | turn #{40 - turns_remaining}]: My position: {current_location}, "
                  f"Egg locations: {sorted(board.eggs_player)}")
        
        # Debug: Final summary when game is ending
        if turns_remaining <= 1:
            corner_eggs = sum(1 for _, is_corner, _, _, _ in self.eggs_laid_log if is_corner)
            regular_eggs = len(self.eggs_laid_log) - corner_eggs
            print(f"[MyAgent | Player {self.player_label} | FINAL SUMMARY]: Total eggs laid: {current_egg_count}")
            print(f"[MyAgent | Player {self.player_label} | FINAL SUMMARY]: Physical eggs: {len(self.eggs_laid_log)} "
                  f"({corner_eggs} corner, {regular_eggs} regular)")
            print(f"[MyAgent | Player {self.player_label} | FINAL SUMMARY]: Corner eggs worth: {corner_eggs * 3}, "
                  f"Regular eggs worth: {regular_eggs}, Total: {corner_eggs * 3 + regular_eggs}")
            print(f"[MyAgent | Player {self.player_label} | FINAL SUMMARY]: Eggs in board set: {len(board.eggs_player)}")
            print(f"[MyAgent | Player {self.player_label} | FINAL SUMMARY]: All egg locations: {sorted(board.eggs_player)}")
        
        self.moves_made += 1
        self.last_egg_count = current_egg_count
        return move

