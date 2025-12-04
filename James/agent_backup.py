from collections.abc import Callable
from typing import List, Tuple, Optional

from game import board, enums
from game.game_map import prob_hear, prob_feel

"""
Cristiano uses the minimax algorithm with alpha-beta pruning to play the chicken game.
It aggressively lays eggs and strategically places turds to block opponents.
PRIORITIZES moving to areas with fewer eggs to maximize egg-laying opportunities!
"""


class PlayerAgent:
    """
    Minimax agent with alpha-beta pruning for the chicken game.
    Prioritizes egg laying and moving to areas with fewer eggs.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        """
        Initialize the minimax agent.
        
        Parameters:
            board: The initial game board state
            time_left: Function that returns remaining time in seconds
        """
        self.time_left = time_left
        self.max_depth = 4  # Default depth, will be adjusted based on time
        self.nodes_evaluated = 0
        # Track suspected trapdoor locations based on sensor data
        self.suspected_trapdoors = set()  # Set of (x, y) positions that might have trapdoors
        # Track visited positions to encourage exploration and avoid getting stuck
        self.visited_positions = set()
        self.visited_positions.add(board.chicken_player.get_location())
        # Track which regions we've explored (left vs right side)
        self.explored_regions = set()  # Track which side of board we've been to
        # Track egg-laying squares we've visited (where we can lay eggs)
        self.visited_egg_squares = set()  # Squares where we've laid eggs or visited
        # Track all possible egg-laying squares on the board
        self.all_egg_squares = self._find_all_egg_squares(board)
        # CRITICAL: Memory of sensor data from each position
        # Maps (x, y) -> [(heard_white, felt_white), (heard_black, felt_black)]
        self.sensor_memory = {}
        # Probability map for trapdoor locations
        # Maps (x, y) -> probability (0.0 to 1.0) that a trapdoor is there
        self.trapdoor_probability = {}
        # Separate probability grids for even and odd parity trapdoors
        # This allows proper normalization and constraint handling
        self.p_even = {}  # Probability map for even squares (x+y is even)
        self.p_odd = {}   # Probability map for odd squares (x+y is odd)
        self._init_trapdoor_priors()
        # Crash detection: track spawn position and last attempted move
        self.spawn_position = board.chicken_player.get_location()
        self.last_attempted_move_target = None  # Where we tried to move last turn
        self.turn_number = 0
    
    def _init_trapdoor_priors(self):
        """
        Initialize prior probabilities for trapdoor locations.
        Trapdoors are more likely in the center of the board (not on edges).
        """
        # Initialize priors based on trapdoor generation rules
        # Edges (0-1, 6-7) have weight 0, middle (2-5) has weight 1, center (3-4) has weight 2
        total_even_weight = 0.0
        total_odd_weight = 0.0
        
        for x in range(8):
            for y in range(8):
                pos = (x, y)
                parity = (x + y) % 2
                
                # Calculate weight based on position
                if 2 <= x <= 5 and 2 <= y <= 5:
                    if 3 <= x <= 4 and 3 <= y <= 4:
                        weight = 2.0  # Center
                    else:
                        weight = 1.0  # Middle
                else:
                    weight = 0.0  # Edge
                
                if parity == 0:
                    self.p_even[pos] = weight
                    total_even_weight += weight
                else:
                    self.p_odd[pos] = weight
                    total_odd_weight += weight
        
        # Normalize
        if total_even_weight > 0:
            for pos in self.p_even:
                self.p_even[pos] /= total_even_weight
        if total_odd_weight > 0:
            for pos in self.p_odd:
                self.p_odd[pos] /= total_odd_weight
    
    def _get_position_after_move(self, current_pos: Tuple[int, int], direction: enums.Direction) -> Tuple[int, int]:
        """Helper to calculate position after a move."""
        x, y = current_pos
        if direction == enums.Direction.UP:
            return (x, y - 1)
        elif direction == enums.Direction.DOWN:
            return (x, y + 1)
        elif direction == enums.Direction.LEFT:
            return (x - 1, y)
        elif direction == enums.Direction.RIGHT:
            return (x + 1, y)
        return current_pos
    
    def _count_nearby_eggs(self, pos: Tuple[int, int], board_state: board.Board, radius: int = 2) -> int:
        """
        Count eggs (both player and enemy) near a position.
        Fewer eggs nearby = more opportunities to lay eggs!
        """
        count = 0
        for egg_pos in board_state.eggs_player | board_state.eggs_enemy:
            dist = abs(egg_pos[0] - pos[0]) + abs(egg_pos[1] - pos[1])
            if dist <= radius:
                count += 1
        return count
    
    def _can_lay_egg_at_position(self, pos: Tuple[int, int], board_state: board.Board) -> bool:
        """Check if we can lay an egg at a given position."""
        # Check if position matches our parity
        if (pos[0] + pos[1]) % 2 != board_state.chicken_player.even_chicken:
            return False
        # Check if position is already occupied
        if pos in board_state.eggs_player or pos in board_state.eggs_enemy:
            return False
        if pos in board_state.turds_player or pos in board_state.turds_enemy:
            return False
        # Check if position is blocked by enemy turd zone
        if board_state.is_cell_in_enemy_turd_zone(pos):
            return False
        return True
    
    def _is_suspected_trapdoor(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is a suspected trapdoor."""
        return pos in self.suspected_trapdoors
    
    def _get_region(self, pos: Tuple[int, int]) -> str:
        """Get which region of the board a position is in (left, right, center)."""
        x, y = pos
        if x <= 2:
            return "left"
        elif x >= 5:
            return "right"
        else:
            return "center"
    
    def _is_unexplored_region(self, pos: Tuple[int, int]) -> bool:
        """Check if we haven't explored this region yet."""
        region = self._get_region(pos)
        return region not in self.explored_regions
    
    def _distance_to_nearest_visited(self, pos: Tuple[int, int]) -> int:
        """Calculate distance to nearest visited position. Higher = more exploration."""
        if not self.visited_positions:
            return 10  # All unexplored
        return min(abs(pos[0] - v[0]) + abs(pos[1] - v[1]) for v in self.visited_positions)
    
    def _can_reach_other_side(self, current_pos: Tuple[int, int], board_state: board.Board) -> bool:
        """Check if we can potentially reach the other side of the board."""
        # Check if there's a path to the other side (not blocked by trapdoors/turds)
        current_region = self._get_region(current_pos)
        target_x = 0 if current_region == "right" else 7
        
        # Simple check: can we move toward the other side?
        for y in range(8):
            test_pos = (target_x, y)
            if board_state.is_valid_cell(test_pos):
                # Check if path is relatively clear
                if not board_state.is_cell_blocked(test_pos):
                    return True
        return False
    
    def _find_all_egg_squares(self, board_state: board.Board) -> set:
        """Find all squares where we can potentially lay eggs."""
        egg_squares = set()
        even_chicken = board_state.chicken_player.even_chicken
        
        for x in range(8):
            for y in range(8):
                pos = (x, y)
                # Check if this square matches our parity (where we can lay eggs)
                if (x + y) % 2 == even_chicken:
                    egg_squares.add(pos)
        
        return egg_squares
    
    def _get_unvisited_egg_squares(self, board_state: board.Board) -> set:
        """Get egg-laying squares we haven't visited yet."""
        unvisited = set()
        for pos in self.all_egg_squares:
            # Check if we can still lay an egg here (not blocked, not already has egg/turd)
            if self._can_lay_egg_at_position(pos, board_state):
                if pos not in self.visited_egg_squares:
                    unvisited.add(pos)
        return unvisited
    
    def _distance_to_nearest_unvisited_egg_square(self, pos: Tuple[int, int], board_state: board.Board) -> int:
        """Calculate distance to nearest unvisited egg-laying square."""
        unvisited = self._get_unvisited_egg_squares(board_state)
        if not unvisited:
            return 0  # All visited
        return min(abs(pos[0] - e[0]) + abs(pos[1] - e[1]) for e in unvisited)
    
    def _find_path_to_unvisited_egg_square(self, start_pos: Tuple[int, int], board_state: board.Board, max_depth: int = 3) -> Optional[Tuple[int, int]]:
        """Find the best next position to move toward unvisited egg squares using BFS."""
        unvisited = self._get_unvisited_egg_squares(board_state)
        if not unvisited:
            return None
        
        # Use BFS to find shortest path to any unvisited egg square
        from collections import deque
        queue = deque([(start_pos, [])])
        visited_bfs = {start_pos}
        
        while queue and len(queue[0][1]) < max_depth:
            current, path = queue.popleft()
            
            # Check all 4 directions
            for direction in [enums.Direction.UP, enums.Direction.DOWN, enums.Direction.LEFT, enums.Direction.RIGHT]:
                next_pos = self._get_position_after_move(current, direction)
                
                # Skip invalid positions
                if not board_state.is_valid_cell(next_pos):
                    continue
                
                # Skip trapdoors
                if next_pos in board_state.found_trapdoors or self._is_suspected_trapdoor(next_pos):
                    continue
                
                # Skip blocked positions
                if board_state.is_cell_blocked(next_pos):
                    continue
                
                # If we've seen this position in BFS, skip
                if next_pos in visited_bfs:
                    continue
                
                visited_bfs.add(next_pos)
                new_path = path + [next_pos]
                
                # If this is an unvisited egg square, return the first step!
                if next_pos in unvisited:
                    return new_path[0] if new_path else next_pos
                
                # Continue searching
                queue.append((next_pos, new_path))
        
        # If no path found, return closest unvisited egg square direction
        closest = min(unvisited, key=lambda e: abs(start_pos[0] - e[0]) + abs(start_pos[1] - e[1]))
        # Find direction that gets us closer
        dx = closest[0] - start_pos[0]
        dy = closest[1] - start_pos[1]
        
        if abs(dx) > abs(dy):
            return (start_pos[0] + (1 if dx > 0 else -1), start_pos[1])
        else:
            return (start_pos[0], start_pos[1] + (1 if dy > 0 else -1))
    
    def _count_reachable_unvisited_egg_squares(self, pos: Tuple[int, int], board_state: board.Board, max_steps: int = 5) -> int:
        """Count how many unvisited egg squares are reachable from this position."""
        unvisited = self._get_unvisited_egg_squares(board_state)
        if not unvisited:
            return 0
        
        # Use BFS to count reachable squares
        from collections import deque
        queue = deque([(pos, 0)])
        visited_bfs = {pos}
        reachable = set()
        
        while queue:
            current, steps = queue.popleft()
            
            if steps >= max_steps:
                continue
            
            # Check if this is an unvisited egg square
            if current in unvisited:
                reachable.add(current)
            
            # Check all 4 directions
            for direction in [enums.Direction.UP, enums.Direction.DOWN, enums.Direction.LEFT, enums.Direction.RIGHT]:
                next_pos = self._get_position_after_move(current, direction)
                
                if not board_state.is_valid_cell(next_pos):
                    continue
                
                if next_pos in board_state.found_trapdoors or self._is_suspected_trapdoor(next_pos):
                    continue
                
                if board_state.is_cell_blocked(next_pos):
                    continue
                
                if next_pos in visited_bfs:
                    continue
                
                visited_bfs.add(next_pos)
                queue.append((next_pos, steps + 1))
        
        return len(reachable)
    
    def _update_trapdoor_suspicions(self, current_pos: Tuple[int, int], sensor_data: List[Tuple[bool, bool]], board_state: board.Board):
        """
        Update suspected trapdoor locations based on sensor data.
        Uses proper Bayesian inference with hard constraints and normalization.
        Sensor data: [(heard_white, felt_white), (heard_black, felt_black)]
        - sensor_data[0] = trapdoor on WHITE squares (i+j is even)
        - sensor_data[1] = trapdoor on BLACK squares (i+j is odd)
        """
        # Store sensor data for this position
        self.sensor_memory[current_pos] = sensor_data
        
        known_trapdoors = board_state.found_trapdoors
        
        # --- STEP 1: SENSORY UPDATE using proper probability functions ---
        # Apply Bayesian update FIRST to incorporate new evidence
        # The grids persist between calls, so evidence accumulates over time
        # Each new observation multiplies the current probabilities (correct Bayesian updating)
        # Apply Bayesian update using ONLY the CURRENT sensor reading
        # The grids persist between calls, so evidence accumulates over time
        # Each new observation multiplies the current probabilities (correct Bayesian updating)
        
        for trapdoor_idx, (heard, felt) in enumerate(sensor_data):
            # trapdoor_idx 0 = even (white), trapdoor_idx 1 = odd (black)
            target_grid = self.p_even if trapdoor_idx == 0 else self.p_odd
            
            # Update probabilities using Bayesian inference with proper likelihoods
            # Only apply the CURRENT sensor reading - grids persist and accumulate evidence
            for pos in list(target_grid.keys()):
                if pos in known_trapdoors:
                    continue  # Skip known trapdoors
                
                # Skip positions that are already marked as safe (probability 0)
                if target_grid[pos] == 0.0:
                    continue
                
                # Calculate delta (non-negative distances)
                dx = abs(pos[0] - current_pos[0])
                dy = abs(pos[1] - current_pos[1])
                
                # Get actual probabilities from game functions
                p_h = prob_hear(dx, dy)
                p_f = prob_feel(dx, dy)
                
                # Calculate likelihood: P(observation | trapdoor at pos)
                # If we heard, likelihood is p_h; if not, likelihood is (1 - p_h)
                l_h = p_h if heard else (1.0 - p_h)
                l_f = p_f if felt else (1.0 - p_f)
                
                # Update probability: P(trapdoor|obs) ∝ P(obs|trapdoor) * P(trapdoor)
                # Multiply current probability by likelihood (Bayesian update)
                target_grid[pos] *= (l_h * l_f)
        
        # --- STEP 2: HARD CONSTRAINTS - Items on board are SAFE ---
        # Squares with eggs, turds, or chickens CANNOT have trapdoors
        # Apply AFTER sensor update so we incorporate evidence first
        safe_squares = set()
        safe_squares.update(board_state.eggs_player)
        safe_squares.update(board_state.eggs_enemy)
        safe_squares.update(board_state.turds_player)
        safe_squares.update(board_state.turds_enemy)
        safe_squares.add(board_state.chicken_player.get_location())
        safe_squares.add(board_state.chicken_enemy.get_location())
        
        for safe_pos in safe_squares:
            if safe_pos in known_trapdoors:
                continue  # Skip if it's a known trapdoor
            parity = (safe_pos[0] + safe_pos[1]) % 2
            # Mark as safe (probability 0.0) - position should already be in grid from initialization
            if parity == 0:
                self.p_even[safe_pos] = 0.0
            else:
                self.p_odd[safe_pos] = 0.0
        
        # --- STEP 3: GLOBAL CONSTRAINTS - Found trapdoor means others of same parity are safe ---
        # Apply AFTER sensor update and hard constraints
        for trapdoor_pos in known_trapdoors:
            parity = (trapdoor_pos[0] + trapdoor_pos[1]) % 2
            if parity == 0:
                # Found even trapdoor - all other even squares are safe
                for pos in list(self.p_even.keys()):
                    if pos != trapdoor_pos:
                        self.p_even[pos] = 0.0
                self.p_even[trapdoor_pos] = 1.0
            else:
                # Found odd trapdoor - all other odd squares are safe
                for pos in list(self.p_odd.keys()):
                    if pos != trapdoor_pos:
                        self.p_odd[pos] = 0.0
                self.p_odd[trapdoor_pos] = 1.0
        
        # --- STEP 4: NORMALIZE probabilities ---
        # Normalize AFTER all updates and constraints
        self._normalize_probabilities()
        
        # --- STEP 5: Update unified probability map and suspected set ---
        self.suspected_trapdoors.clear()
        for pos in known_trapdoors:
            self.trapdoor_probability[pos] = 1.0
            self.suspected_trapdoors.add(pos)
        
        # Combine even and odd probabilities into unified map
        for pos, prob in self.p_even.items():
            if pos not in known_trapdoors:
                self.trapdoor_probability[pos] = prob
                if prob > 0.3:
                    self.suspected_trapdoors.add(pos)
        
        for pos, prob in self.p_odd.items():
            if pos not in known_trapdoors:
                self.trapdoor_probability[pos] = prob
                if prob > 0.3:
                    self.suspected_trapdoors.add(pos)
    
    def _normalize_probabilities(self):
        """Normalize probability distributions so they sum to 1.0 for each parity."""
        # Normalize even probabilities
        total_even = sum(self.p_even.values())
        if total_even > 0:
            for pos in self.p_even:
                self.p_even[pos] /= total_even
        
        # Normalize odd probabilities
        total_odd = sum(self.p_odd.values())
        if total_odd > 0:
            for pos in self.p_odd:
                self.p_odd[pos] /= total_odd
    
    def _get_trapdoor_probability(self, pos: Tuple[int, int]) -> float:
        """Get the probability that a trapdoor is at this position."""
        # Check unified map first
        if pos in self.trapdoor_probability:
            return self.trapdoor_probability[pos]
        
        # Fall back to parity-specific maps
        parity = (pos[0] + pos[1]) % 2
        if parity == 0:
            return self.p_even.get(pos, 0.0)
        else:
            return self.p_odd.get(pos, 0.0)
    
    def _is_likely_trapdoor(self, pos: Tuple[int, int], threshold: float = 0.3) -> bool:
        """Check if a position is likely to be a trapdoor."""
        return self._get_trapdoor_probability(pos) >= threshold
        
    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Play a move using minimax with alpha-beta pruning.
        
        Parameters:
            board: Current game board state
            sensor_data: List of (heard, felt) tuples for trapdoors
            time_left: Function that returns remaining time in seconds
            
        Returns:
            Tuple of (Direction, MoveType) representing the chosen move
        """
        self.time_left = time_left
        self.nodes_evaluated = 0
        self.turn_number += 1
        
        # CRASH DETECTION: Detect if we fell into a trapdoor last turn
        current_pos = board.chicken_player.get_location()
        if self.last_attempted_move_target is not None:
            # If we're back at spawn but we tried to move somewhere else, we fell into a trapdoor
            if current_pos == self.spawn_position and self.last_attempted_move_target != self.spawn_position:
                # The square we tried to step onto is the trapdoor!
                trapdoor_pos = self.last_attempted_move_target
                # Mark it as a known trapdoor (will be in board.found_trapdoors, but this helps immediately)
                # Also update our probability maps
                parity = (trapdoor_pos[0] + trapdoor_pos[1]) % 2
                if parity == 0:
                    # Found even trapdoor - all other even squares are safe
                    for pos in list(self.p_even.keys()):
                        if pos != trapdoor_pos:
                            self.p_even[pos] = 0.0
                    self.p_even[trapdoor_pos] = 1.0
                else:
                    # Found odd trapdoor - all other odd squares are safe
                    for pos in list(self.p_odd.keys()):
                        if pos != trapdoor_pos:
                            self.p_odd[pos] = 0.0
                    self.p_odd[trapdoor_pos] = 1.0
                # Update unified probability map
                self.trapdoor_probability[trapdoor_pos] = 1.0
                self.suspected_trapdoors.add(trapdoor_pos)
        
        # Update trapdoor suspicions based on sensor data
        self._update_trapdoor_suspicions(current_pos, sensor_data, board)
        
        # Track visited positions and regions for exploration
        self.visited_positions.add(current_pos)
        current_region = self._get_region(current_pos)
        self.explored_regions.add(current_region)
        
        # Track egg-laying squares we've visited
        if self._can_lay_egg_at_position(current_pos, board):
            self.visited_egg_squares.add(current_pos)
        # Also mark squares where we've laid eggs
        self.visited_egg_squares.update(board.eggs_player)
        
        # Adjust depth based on remaining time and game state
        remaining_time = time_left()
        turns_left = board.turns_left_player
        
        # Use deeper search if we have more time
        # Prioritize depth to find best egg-laying and turd-blocking moves
        if remaining_time > 300:  # More than 5 minutes left
            self.max_depth = 6
        elif remaining_time > 180:  # More than 3 minutes left
            self.max_depth = 5
        elif remaining_time > 60:  # More than 1 minute left
            self.max_depth = 4
        elif remaining_time > 20:  # More than 20 seconds left
            self.max_depth = 3
        else:  # Less than 20 seconds
            self.max_depth = 2
        
        # If we're near the end of the game, search deeper to maximize eggs
        if turns_left <= 10:
            self.max_depth = min(self.max_depth + 1, 7)
        elif turns_left <= 20:
            self.max_depth = min(self.max_depth + 1, 6)
        
        # Get valid moves
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            # No valid moves (shouldn't happen, but handle gracefully)
            return (enums.Direction.UP, enums.MoveType.PLAIN)
        
        # CRITICAL: Filter out moves that would step on trapdoors!
        # This is ABSOLUTE PRIORITY - stepping on trapdoor gives opponent 4 eggs!
        safe_moves = []
        for move in valid_moves:
            dir, move_type = move
            # Calculate destination position (where we end up after the move)
            # For all move types, we move to a new position
            new_pos = self._get_position_after_move(current_pos, dir)
            
            # NEVER step on known trapdoors!
            if new_pos in board.found_trapdoors:
                continue  # Skip this move - it's a known trapdoor!
            
            # Avoid likely trapdoors based on probability
            trapdoor_prob = self._get_trapdoor_probability(new_pos)
            if trapdoor_prob > 0.5:  # High probability - avoid!
                continue  # Skip this move - very likely a trapdoor!
            elif trapdoor_prob > 0.3:  # Medium probability - strongly penalize but don't eliminate
                # We'll penalize these in scoring, but keep as last resort
                pass
            
            safe_moves.append(move)
        
        # If no safe moves after filtering, use moves with lower trapdoor probability
        if not safe_moves:
            # Emergency: all moves might lead to trapdoors
            # Sort by trapdoor probability (lowest first)
            def trapdoor_risk(move):
                dir, move_type = move
                new_pos = self._get_position_after_move(current_pos, dir)
                return self._get_trapdoor_probability(new_pos)
            
            valid_moves = sorted(valid_moves, key=trapdoor_risk)
            safe_moves = [m for m in valid_moves if self._get_trapdoor_probability(
                self._get_position_after_move(current_pos, m[0])
            ) < 0.7]  # Only use moves with < 70% trapdoor probability
            
            if not safe_moves:
                # Worst case - use least risky move
                safe_moves = [valid_moves[0]]
        
        valid_moves = safe_moves
        
        # If only one move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # CRITICAL: Check if we're currently on a square where we can lay an egg!
        current_pos = board.chicken_player.get_location()
        can_lay_egg_now = board.can_lay_egg()
        enemy_pos = board.chicken_enemy.get_location()
        dist_to_enemy = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
        
        # If we can lay an egg RIGHT NOW, prioritize it (but still consider turds if opponent is adjacent)
        if can_lay_egg_now:
            egg_moves = [m for m in valid_moves if m[1] == enums.MoveType.EGG]
            if egg_moves:
                # We're on an egg-laying square - prioritize egg moves!
                # But if opponent is adjacent, also consider turds as an option
                if dist_to_enemy == 1:
                    turd_moves = [m for m in valid_moves if m[1] == enums.MoveType.TURD]
                    if turd_moves:
                        # Both eggs and turds available - prioritize eggs but consider turds
                        valid_moves = egg_moves + turd_moves
                    else:
                        valid_moves = egg_moves
                else:
                    # Only egg moves - opponent not adjacent
                    valid_moves = egg_moves
            # If no egg moves available (shouldn't happen if can_lay_egg_now is true), continue
        
        # AGGRESSIVE EGG LAYING: If we can lay an egg, strongly prioritize it
        egg_moves = [m for m in valid_moves if m[1] == enums.MoveType.EGG]
        
        # TURD STRATEGY: Only lay turds if opponent is ADJACENT (1 square away)!
        enemy_pos = board.chicken_enemy.get_location()
        dist_to_enemy = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
        
        # Filter turd moves - only keep them if opponent is ADJACENT (1 square away)
        turd_moves = [m for m in valid_moves if m[1] == enums.MoveType.TURD]
        if dist_to_enemy != 1:
            # Opponent is NOT adjacent - remove turd moves, focus on eggs and exploration
            valid_moves = [m for m in valid_moves if m[1] != enums.MoveType.TURD]
        
        # EXPLORATION CHECK: If we're stuck in one region, prioritize moves to other regions
        current_region = self._get_region(current_pos)
        if len(self.explored_regions) == 1:
            # We've only explored one region - STRONGLY prioritize moving to other regions!
            moves_to_other_side = []
            for move in valid_moves:
                dir, move_type = move
                if move_type == enums.MoveType.PLAIN:
                    new_pos = self._get_position_after_move(current_pos, dir)
                    new_region = self._get_region(new_pos)
                    if new_region != current_region:
                        moves_to_other_side.append(move)
            if moves_to_other_side:
                # Prioritize moves that get us to the other side!
                valid_moves = moves_to_other_side + [m for m in valid_moves if m not in moves_to_other_side]
        
        # If we've been in the same area too much, encourage exploration
        if len(self.visited_positions) >= 5:
            recent_visits_nearby = sum(
                1 for v in list(self.visited_positions)[-10:]
                if abs(v[0] - current_pos[0]) + abs(v[1] - current_pos[1]) <= 2
            )
            if recent_visits_nearby >= 3:
                # We're stuck! Prioritize exploration moves
                exploration_moves = []
                for move in valid_moves:
                    dir, move_type = move
                    if move_type == enums.MoveType.PLAIN:
                        new_pos = self._get_position_after_move(current_pos, dir)
                        dist_to_visited = self._distance_to_nearest_visited(new_pos)
                        if dist_to_visited >= 3 or new_pos not in self.visited_positions:
                            exploration_moves.append(move)
                if exploration_moves:
                    valid_moves = exploration_moves + [m for m in valid_moves if m not in exploration_moves]
        
        if egg_moves:
            # If we have egg moves available, prioritize them heavily
            # Still use minimax but with egg moves first
            valid_moves = egg_moves + [m for m in valid_moves if m[1] != enums.MoveType.EGG]
        
        # DOUBLE CHECK: If we're on an egg-laying square and have egg moves, ONLY use egg moves!
        if can_lay_egg_now and egg_moves:
            valid_moves = egg_moves  # Force egg laying when we're on an egg square!
        
        # Use minimax with alpha-beta pruning to find best move
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Sort moves by heuristic to improve alpha-beta pruning efficiency
        # HEAVILY prioritize egg moves, then strategic turd moves, then plain moves
        # For plain moves, prioritize moving to areas with FEWER eggs!
        def move_score(move):
            dir, move_type = move
            current_pos = board.chicken_player.get_location()
            enemy_pos = board.chicken_enemy.get_location()
            
            if move_type == enums.MoveType.EGG:
                # Egg is laid at CURRENT position (before moving)
                # CRITICAL: If we're on an egg-laying square, eggs are MANDATORY!
                if can_lay_egg_now:
                    # We're on an egg square - eggs are TOP priority!
                    is_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
                    return 100000 if is_corner else 50000  # MASSIVE priority for eggs when we can lay them!
                # Normal egg priority
                is_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
                return 1000 if is_corner else 100  # Corner eggs are super valuable
            elif move_type == enums.MoveType.PLAIN:
                # STRATEGIC TRAVERSAL: Move efficiently to visit all egg-laying squares!
                new_pos = self._get_position_after_move(current_pos, dir)
                
                # CRITICAL: Avoid trapdoors based on probability!
                if new_pos in board.found_trapdoors:
                    return -100000  # NEVER step on known trapdoors!
                
                trapdoor_prob = self._get_trapdoor_probability(new_pos)
                if trapdoor_prob > 0.7:
                    return -50000  # Very high probability - avoid at all costs!
                elif trapdoor_prob > 0.5:
                    return -20000  # High probability - strongly avoid!
                elif trapdoor_prob > 0.3:
                    return -10000  # Medium probability - avoid!
                elif self._is_suspected_trapdoor(new_pos):
                    return -5000  # Suspected trapdoor - avoid!
                
                # Check if this is an unvisited egg-laying square
                can_lay_egg_there = self._can_lay_egg_at_position(new_pos, board)
                is_unvisited_egg_square = can_lay_egg_there and new_pos not in self.visited_egg_squares
                
                # Calculate strategic metrics
                dist_to_unvisited_egg = self._distance_to_nearest_unvisited_egg_square(new_pos, board)
                reachable_from_here = self._count_reachable_unvisited_egg_squares(new_pos, board, max_steps=5)
                target_pos = self._find_path_to_unvisited_egg_square(new_pos, board, max_depth=3)
                is_on_path_to_egg = target_pos is not None
                
                score = 10  # Base score
                
                # MASSIVE PRIORITY: Move to unvisited egg-laying squares!
                if is_unvisited_egg_square:
                    score += 20000  # HUGE bonus for unvisited egg squares!
                    # Extra bonus if it's a corner (worth 3 eggs!)
                    if (new_pos[0] == 0 or new_pos[0] == 7) and (new_pos[1] == 0 or new_pos[1] == 7):
                        score += 10000  # Corner egg square is SUPER valuable!
                elif can_lay_egg_there:
                    # We can lay an egg here but already visited - still good
                    score += 100  # Can lay egg here
                    if (new_pos[0] == 0 or new_pos[0] == 7) and (new_pos[1] == 0 or new_pos[1] == 7):
                        score += 50  # Corner opportunity
                
                # STRATEGIC BONUS: Positions that give access to many unvisited egg squares
                if reachable_from_here > 0:
                    score += reachable_from_here * 500  # More reachable squares = better position!
                    if reachable_from_here >= 3:
                        score += 2000  # Excellent position - can reach many squares!
                
                # BONUS for being on a path to unvisited egg squares
                if is_on_path_to_egg:
                    score += 3000  # This move is on the path to an egg square!
                
                # BONUS for moving closer to unvisited egg squares
                if dist_to_unvisited_egg > 0:
                    # Closer is better, but also consider how many we can reach
                    if dist_to_unvisited_egg <= 1:
                        score += 5000  # Very close to unvisited egg square!
                    elif dist_to_unvisited_egg <= 2:
                        score += 3000  # Close to unvisited egg square!
                    elif dist_to_unvisited_egg <= 4:
                        score += 1500  # Getting closer to unvisited egg square
                    elif dist_to_unvisited_egg <= 6:
                        score += 800  # Moving toward unvisited egg squares
                
                # Exploration bonuses (still important for navigation)
                new_region = self._get_region(new_pos)
                current_region = self._get_region(current_pos)
                is_new_region = new_region != current_region and self._is_unexplored_region(new_pos)
                is_unexplored = new_pos not in self.visited_positions
                
                if is_new_region:
                    score += 2000  # Bonus for new region
                elif is_unexplored:
                    score += 500  # Bonus for unexplored area
                
                # Count eggs nearby (less important than visiting egg squares)
                nearby_eggs = self._count_nearby_eggs(new_pos, board, radius=2)
                if nearby_eggs == 0:
                    score += 20
                elif nearby_eggs >= 4:
                    score -= 10
                
                return score
            elif move_type == enums.MoveType.TURD:
                # Turd is placed at current position
                # ONLY lay turds if opponent is ADJACENT (1 square away)!
                turd_pos = current_pos
                dist_to_enemy = abs(turd_pos[0] - enemy_pos[0]) + abs(turd_pos[1] - enemy_pos[1])
                if dist_to_enemy == 1:
                    return 200  # Opponent is adjacent - lay turd to block them!
                else:
                    return -10000  # Opponent is NOT adjacent - don't waste turds!
            else:
                return 1  # Plain moves are lowest priority
        
        sorted_moves = sorted(valid_moves, key=move_score, reverse=True)
        
        for move in sorted_moves:
            if time_left() < 0.1:  # Stop if running out of time
                break
                
            dir, move_type = move
            forecasted_board = board.forecast_move(dir, move_type, check_ok=False)
            
            if forecasted_board is None:
                continue
            
            # After forecast_move, the turn has switched to opponent
            # Reverse perspective so opponent is now "player" from board's perspective
            forecasted_board.reverse_perspective()
            
            # Evaluate this move (opponent's turn next, so minimizing)
            value = self.minimax(
                forecasted_board,
                self.max_depth - 1,
                alpha,
                beta,
                maximizing=False,  # Opponent's turn
            )
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, best_value)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break
        
        # Fallback to first valid move if something went wrong
        if best_move is None:
            best_move = valid_moves[0]
        
        # Track where we're trying to move (for crash detection)
        dir, move_type = best_move
        if move_type == enums.MoveType.PLAIN:
            # For plain moves, track the destination
            self.last_attempted_move_target = self._get_position_after_move(current_pos, dir)
        else:
            # For egg/turd moves, we stay in place, so no target to track
            self.last_attempted_move_target = current_pos
        
        return best_move
    
    def minimax(
        self,
        board_state: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Parameters:
            board_state: Current board state
            depth: Remaining search depth
            alpha: Best value for maximizing player
            beta: Best value for minimizing player
            maximizing: True if it's the maximizing player's turn
            
        Returns:
            Evaluation value of the board state
        """
        self.nodes_evaluated += 1
        
        # Check if we're running out of time
        if self.time_left() < 0.05:
            return self.evaluate(board_state)
        
        # Terminal conditions
        if board_state.is_game_over():
            return self.evaluate_terminal(board_state)
        
        if depth == 0:
            return self.evaluate(board_state)
        
        # Get valid moves for current player
        valid_moves = board_state.get_valid_moves()
        
        if not valid_moves:
            # No moves available - this is a terminal state
            return self.evaluate_terminal(board_state)
        
        if maximizing:
            max_eval = float('-inf')
            
            # Check if current player can lay an egg at their position
            current_player_pos = board_state.chicken_player.get_location()
            can_lay_egg_at_pos = board_state.can_lay_egg()
            
            # HEAVILY prioritize egg moves, then strategic turds, then plain moves
            # For plain moves, prioritize moving to areas with fewer eggs!
            def move_score(move):
                dir, move_type = move
                if move_type == enums.MoveType.EGG:
                    # If we can lay an egg, it's MANDATORY!
                    if can_lay_egg_at_pos:
                        is_corner = (current_player_pos[0] == 0 or current_player_pos[0] == 7) and (current_player_pos[1] == 0 or current_player_pos[1] == 7)
                        return 100000 if is_corner else 50000  # MASSIVE priority!
                    return 1000  # Normal egg priority
                elif move_type == enums.MoveType.PLAIN:
                    current_pos = board_state.chicken_player.get_location()
                    new_pos = self._get_position_after_move(current_pos, dir)
                    
                    # CRITICAL: Avoid trapdoors based on probability!
                    if new_pos in board_state.found_trapdoors:
                        return -100000  # NEVER step on known trapdoors!
                    
                    trapdoor_prob = self._get_trapdoor_probability(new_pos)
                    if trapdoor_prob > 0.7:
                        return -50000  # Very high probability - avoid!
                    elif trapdoor_prob > 0.5:
                        return -20000  # High probability - avoid!
                    elif trapdoor_prob > 0.3:
                        return -10000  # Medium probability - avoid!
                    elif self._is_suspected_trapdoor(new_pos):
                        return -5000  # Suspected trapdoor - avoid!
                    
                    # Strategic traversal: prioritize positions that give access to many egg squares
                    can_lay_egg = self._can_lay_egg_at_position(new_pos, board_state)
                    is_unvisited_egg_square = can_lay_egg and new_pos not in self.visited_egg_squares
                    dist_to_unvisited_egg = self._distance_to_nearest_unvisited_egg_square(new_pos, board_state)
                    reachable_from_here = self._count_reachable_unvisited_egg_squares(new_pos, board_state, max_steps=5)
                    target_pos = self._find_path_to_unvisited_egg_square(new_pos, board_state, max_depth=3)
                    is_on_path_to_egg = target_pos is not None
                    
                    score = 10
                    
                    if is_unvisited_egg_square:
                        score += 20000  # HUGE bonus for unvisited egg squares!
                    elif can_lay_egg:
                        score += 100
                    
                    # Strategic positioning: positions that give access to many squares
                    if reachable_from_here > 0:
                        score += reachable_from_here * 500
                        if reachable_from_here >= 3:
                            score += 2000
                    
                    if is_on_path_to_egg:
                        score += 3000
                    
                    if dist_to_unvisited_egg > 0:
                        if dist_to_unvisited_egg <= 1:
                            score += 5000
                        elif dist_to_unvisited_egg <= 2:
                            score += 3000
                        elif dist_to_unvisited_egg <= 4:
                            score += 1500
                    
                    nearby_eggs = self._count_nearby_eggs(new_pos, board_state, radius=2)
                    if nearby_eggs == 0:
                        score += 20
                    elif nearby_eggs >= 4:
                        score -= 10
                    
                    return score
                elif move_type == enums.MoveType.TURD:
                    # ONLY lay turds if opponent is ADJACENT (1 square away)!
                    player_pos = board_state.chicken_player.get_location()
                    enemy_pos = board_state.chicken_enemy.get_location()
                    dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
                    if dist_to_enemy == 1:
                        return 200  # Opponent adjacent - lay turd!
                    else:
                        return -10000  # Opponent NOT adjacent - don't waste turds!
                else:
                    return 1
            
            sorted_moves = sorted(valid_moves, key=move_score, reverse=True)
            
            for move in sorted_moves:
                dir, move_type = move
                forecasted_board = board_state.forecast_move(dir, move_type, check_ok=False)
                
                if forecasted_board is None:
                    continue
                
                # After forecast_move, turn switched to opponent
                # Reverse perspective so opponent is "player" from board's perspective
                forecasted_board.reverse_perspective()
                
                eval_score = self.minimax(
                    forecasted_board,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=False,  # Next is opponent's turn
                )
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_eval
        else:
            min_eval = float('inf')
            
            # For opponent, we want them to make bad moves, so reverse priority
            # (but still need to explore their best moves first for accurate evaluation)
            def move_score(move):
                dir, move_type = move
                if move_type == enums.MoveType.EGG:
                    return 1000  # Opponent laying eggs is bad for us
                elif move_type == enums.MoveType.TURD:
                    return 10
                else:
                    return 1
            
            sorted_moves = sorted(valid_moves, key=move_score, reverse=True)
            
            for move in sorted_moves:
                dir, move_type = move
                forecasted_board = board_state.forecast_move(dir, move_type, check_ok=False)
                
                if forecasted_board is None:
                    continue
                
                # After forecast_move, turn switched to opponent
                # Reverse perspective so opponent is "player" from board's perspective
                forecasted_board.reverse_perspective()
                
                eval_score = self.minimax(
                    forecasted_board,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=True,  # Next is our turn
                )
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return min_eval
    
    def evaluate(self, board_state: board.Board) -> float:
        """
        Evaluate a board state from the current player's perspective.
        Higher values are better for the current player.
        HEAVILY prioritizes egg laying and strategic turd placement.
        
        Parameters:
            board_state: Board state to evaluate
            
        Returns:
            Evaluation score
        """
        # PRIMARY FACTOR: Egg count difference - THIS IS THE MOST IMPORTANT!
        player_eggs = board_state.chicken_player.get_eggs_laid()
        enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        egg_diff = player_eggs - enemy_eggs
        
        # Weight egg difference VERY heavily - eggs win the game!
        # Near endgame, weight even more
        turns_left = board_state.turns_left_player
        if turns_left <= 10:
            egg_weight = 5000  # Endgame: eggs are everything
        elif turns_left <= 20:
            egg_weight = 2000  # Midgame: eggs are very important
        else:
            egg_weight = 1000  # Early game: eggs are important
        
        score = egg_diff * egg_weight
        
        # CORNER EGGS BONUS - Corner eggs are worth 3 instead of 1!
        corner_bonus = 0
        for egg_pos in board_state.eggs_player:
            if (egg_pos[0] == 0 or egg_pos[0] == 7) and (egg_pos[1] == 0 or egg_pos[1] == 7):
                corner_bonus += 2  # Already counted as 1 egg, so add 2 more
        score += corner_bonus * egg_weight  # Corner eggs are super valuable
        
        # Penalty for enemy corner eggs
        enemy_corner_penalty = 0
        for egg_pos in board_state.eggs_enemy:
            if (egg_pos[0] == 0 or egg_pos[0] == 7) and (egg_pos[1] == 0 or egg_pos[1] == 7):
                enemy_corner_penalty += 2
        score -= enemy_corner_penalty * egg_weight
        
        # POSITIONAL BONUS: Reward being in areas with FEWER eggs!
        # This gives us more opportunities to lay eggs!
        player_pos = board_state.chicken_player.get_location()
        enemy_pos = board_state.chicken_enemy.get_location()
        
        # Count eggs near our position
        nearby_eggs_player = self._count_nearby_eggs(player_pos, board_state, radius=2)
        nearby_eggs_enemy = self._count_nearby_eggs(enemy_pos, board_state, radius=2)
        
        # BONUS for being in areas with fewer eggs (more egg-laying opportunities!)
        if nearby_eggs_player == 0:
            score += 500  # Perfect spot - no eggs nearby, lots of opportunities!
        elif nearby_eggs_player == 1:
            score += 300  # Great spot - very few eggs nearby
        elif nearby_eggs_player == 2:
            score += 150  # Good spot - some eggs nearby
        elif nearby_eggs_player >= 4:
            score -= 200  # Bad spot - too many eggs, fewer opportunities
        
        # PRIORITY: Reward being at unvisited egg-laying squares and strategic positions!
        can_lay_egg_here = self._can_lay_egg_at_position(player_pos, board_state)
        is_unvisited_egg_square = can_lay_egg_here and player_pos not in self.visited_egg_squares
        dist_to_unvisited_egg = self._distance_to_nearest_unvisited_egg_square(player_pos, board_state)
        reachable_from_here = self._count_reachable_unvisited_egg_squares(player_pos, board_state, max_steps=5)
        
        # CRITICAL: If we can lay an egg here, we MUST have laid one!
        # Check if we already have an egg here
        has_egg_here = player_pos in board_state.eggs_player
        
        if can_lay_egg_here and not has_egg_here:
            # We're on an egg-laying square but haven't laid an egg - this is VERY BAD!
            # This means we moved here but didn't lay an egg - penalize MASSIVELY!
            score -= 100000  # MASSIVE penalty for being on egg square without laying egg!
        
        if is_unvisited_egg_square and has_egg_here:
            score += 10000  # HUGE bonus for being at unvisited egg square WITH an egg!
            if (player_pos[0] == 0 or player_pos[0] == 7) and (player_pos[1] == 0 or player_pos[1] == 7):
                score += 5000  # Corner egg square is SUPER valuable!
        elif can_lay_egg_here and has_egg_here:
            score += 200  # Can lay egg here and we did (already visited)
        elif can_lay_egg_here:
            # We can lay an egg here but haven't - this is a future opportunity
            score += 400  # We can lay an egg here!
            if (player_pos[0] == 0 or player_pos[0] == 7) and (player_pos[1] == 0 or player_pos[1] == 7):
                score += 300  # Corner egg opportunity!
        
        # STRATEGIC POSITIONING: Reward positions that give access to many unvisited egg squares
        if reachable_from_here > 0:
            score += reachable_from_here * 1000  # More reachable squares = better position!
            if reachable_from_here >= 3:
                score += 5000  # Excellent strategic position!
            elif reachable_from_here >= 2:
                score += 2000  # Good strategic position
        
        # BONUS for being close to unvisited egg squares
        if dist_to_unvisited_egg > 0:
            if dist_to_unvisited_egg <= 1:
                score += 2000  # Very close to unvisited egg square!
            elif dist_to_unvisited_egg <= 2:
                score += 1000  # Close to unvisited egg square!
            elif dist_to_unvisited_egg <= 4:
                score += 500  # Getting close to unvisited egg square
        
        # Exploration bonuses (still useful for navigation)
        current_region = self._get_region(player_pos)
        if self._is_unexplored_region(player_pos):
            score += 1000  # Bonus for new region
        elif player_pos not in self.visited_positions:
            score += 200  # Bonus for unexplored area
        
        # TRAPDOOR AVOIDANCE - Stepping on a trapdoor is VERY BAD!
        # Penalize based on trapdoor probability
        if player_pos in board_state.found_trapdoors:
            score -= 100000  # MASSIVE penalty for being on a known trapdoor!
        else:
            trapdoor_prob = self._get_trapdoor_probability(player_pos)
            if trapdoor_prob > 0.7:
                score -= 50000  # Very high probability - massive penalty!
            elif trapdoor_prob > 0.5:
                score -= 20000  # High probability - large penalty!
            elif trapdoor_prob > 0.3:
                score -= 10000  # Medium probability - penalty!
            elif self._is_suspected_trapdoor(player_pos):
                score -= 5000  # Suspected trapdoor - penalty
        
        # Penalty for being adjacent to trapdoors (dangerous!)
        for trapdoor_pos in board_state.found_trapdoors:
            dist_to_trapdoor = abs(player_pos[0] - trapdoor_pos[0]) + abs(player_pos[1] - trapdoor_pos[1])
            if dist_to_trapdoor == 1:
                score -= 5000  # Being adjacent to trapdoor is dangerous!
            elif dist_to_trapdoor == 2:
                score -= 1000  # Being close to trapdoor is risky
        
        # Also penalize being near likely trapdoors (based on probability)
        for x in range(8):
            for y in range(8):
                pos = (x, y)
                if pos == player_pos:
                    continue
                trapdoor_prob = self._get_trapdoor_probability(pos)
                if trapdoor_prob > 0.5:
                    dist = abs(player_pos[0] - x) + abs(player_pos[1] - y)
                    if dist == 1:
                        score -= 2000  # Adjacent to likely trapdoor
                    elif dist == 2:
                        score -= 500  # Close to likely trapdoor
        
        # Penalty if enemy is in a good spot (few eggs nearby)
        if nearby_eggs_enemy == 0:
            score -= 200  # Enemy is in a good spot
        elif nearby_eggs_enemy >= 4:
            score += 100  # Enemy is in a crowded spot
        
        # Bonus if enemy is near trapdoors (they might step on them!)
        for trapdoor_pos in board_state.found_trapdoors:
            dist_enemy_to_trapdoor = abs(enemy_pos[0] - trapdoor_pos[0]) + abs(enemy_pos[1] - trapdoor_pos[1])
            if dist_enemy_to_trapdoor == 1:
                score += 2000  # Enemy is adjacent to trapdoor - they might step on it!
            elif dist_enemy_to_trapdoor == 2:
                score += 500  # Enemy is close to trapdoor
        
        # STRATEGIC TURD PLACEMENT - Only valuable if opponent is nearby (within 2 squares)!
        # Our turds blocking opponent - ONLY valuable when opponent is close!
        turd_bonus = 0
        for turd_pos in board_state.turds_player:
            dist_to_enemy = abs(turd_pos[0] - enemy_pos[0]) + abs(turd_pos[1] - enemy_pos[1])
            if dist_to_enemy <= 1:
                turd_bonus += 200  # Turd adjacent to enemy - blocks them hard!
            elif dist_to_enemy == 2:
                turd_bonus += 100  # Turd close to enemy - good blocking
            else:
                # Turd is far from enemy - this is wasteful! Penalize it
                turd_bonus -= 100  # Wasted turd - should have explored instead
        
        # Also check if turds block enemy's potential egg-laying squares (only if nearby)
        for turd_pos in board_state.turds_player:
            dist_to_enemy = abs(turd_pos[0] - enemy_pos[0]) + abs(turd_pos[1] - enemy_pos[1])
            if dist_to_enemy <= 2:
                # Check if turd is on a square where enemy could lay eggs
                if (turd_pos[0] + turd_pos[1]) % 2 == board_state.chicken_enemy.even_chicken:
                    turd_bonus += 30  # Blocking enemy's egg square is valuable
        
        score += turd_bonus
        
        # Penalty for enemy turds blocking us
        turd_penalty = 0
        for turd_pos in board_state.turds_enemy:
            dist_to_player = abs(turd_pos[0] - player_pos[0]) + abs(turd_pos[1] - player_pos[1])
            if dist_to_player == 1:
                turd_penalty += 150  # Enemy turd adjacent to us - very bad!
            elif dist_to_player == 2:
                turd_penalty += 75   # Enemy turd close to us - bad
            elif dist_to_player <= 3:
                turd_penalty += 30   # Enemy turd near us - somewhat bad
        
        # Also check if enemy turds block our egg-laying squares
        for turd_pos in board_state.turds_enemy:
            if (turd_pos[0] + turd_pos[1]) % 2 == board_state.chicken_player.even_chicken:
                turd_penalty += 20  # Enemy blocking our egg square is bad
        
        score -= turd_penalty
        
        # MOBILITY - Having more moves is better!
        try:
            player_moves = len(board_state.get_valid_moves())
            enemy_moves = len(board_state.get_valid_moves(enemy=True))
            mobility_diff = player_moves - enemy_moves
            
            # If opponent has no moves, that's GREAT (they get penalized 5 eggs)
            if enemy_moves == 0 and board_state.turns_left_enemy > 0:
                score += 50000  # Massive bonus for blocking opponent completely!
            
            score += mobility_diff * 100  # Mobility is valuable
        except:
            pass  # If mobility check fails, skip it
        
        # Remaining turds - having more turds left means more blocking potential
        turds_left_diff = board_state.chicken_player.get_turds_left() - board_state.chicken_enemy.get_turds_left()
        score += turds_left_diff * 50  # Turds are valuable for blocking
        
        # Turn advantage - having more turns means more egg-laying opportunities
        turn_diff = board_state.turns_left_player - board_state.turns_left_enemy
        score += turn_diff * 10
        
        return score
    
    def evaluate_terminal(self, board_state: board.Board) -> float:
        """
        Evaluate a terminal board state.
        Uses egg counts directly to avoid perspective issues.
        
        Parameters:
            board_state: Terminal board state
            
        Returns:
            Large positive value if player wins, large negative if player loses
        """
        # Use egg counts directly to avoid perspective reversal issues
        player_eggs = board_state.chicken_player.get_eggs_laid()
        enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        
        if player_eggs > enemy_eggs:
            return 10000  # Current player wins
        elif player_eggs < enemy_eggs:
            return -10000  # Current enemy wins
        else:
            # Tie - return small value based on other factors
            return (player_eggs - enemy_eggs) * 1000

