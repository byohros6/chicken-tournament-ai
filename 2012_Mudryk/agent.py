import math
import random
import numpy as np
from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
from game import board, enums
from game.enums import Direction, MoveType, loc_after_direction
from game.game_map import prob_hear, prob_feel


class TrapdoorOracle:
    """
    Bayesian inference system for tracking trapdoor locations based on sensor data.
    Stores sensor data from all visited squares and uses it to estimate trapdoor probabilities.
    """
    
    def __init__(self, map_size=8):
        self.map_size = map_size
        # Separate belief grids for white (even) and black (odd) trapdoors
        self.belief_white = np.zeros((map_size, map_size))
        self.belief_black = np.zeros((map_size, map_size))
        # Store all sensor readings: (position) -> [(heard_white, felt_white), (heard_black, felt_black)]
        self.sensor_readings = {}
        # Known trapdoors (stepped on)
        self.known_trapdoors = set()
        
        self._initialize_priors()
    
    def _initialize_priors(self):
        """Initialize prior probabilities based on trapdoor placement rules."""
        dim = self.map_size
        weights = np.zeros((dim, dim))
        
        # Squares on edge have weight 0
        # Squares inside those are weight 0
        # Squares inside those are weight 1
        # Squares inside those are weight 2
        # This matches the trapdoor_manager.py logic
        weights[2:dim-2, 2:dim-2] = 1.0
        weights[3:dim-3, 3:dim-3] = 2.0
        
        # Normalize separately for white and black squares
        for r in range(dim):
            for c in range(dim):
                if (r + c) % 2 == 0:  # White square
                    self.belief_white[r, c] = weights[r, c]
                else:  # Black square
                    self.belief_black[r, c] = weights[r, c]
        
        # Normalize to probabilities
        sum_white = np.sum(self.belief_white)
        sum_black = np.sum(self.belief_black)
        if sum_white > 0:
            self.belief_white /= sum_white
        if sum_black > 0:
            self.belief_black /= sum_black
    
    def add_known_trapdoor(self, pos: Tuple[int, int]):
        """Mark a trapdoor as known (we stepped on it)."""
        self.known_trapdoors.add(pos)
        r, c = pos
        # Set probability to 1.0 for known trapdoor, 0.0 for others of same color
        if (r + c) % 2 == 0:
            self.belief_white.fill(0.0)
            self.belief_white[r, c] = 1.0
        else:
            self.belief_black.fill(0.0)
            self.belief_black[r, c] = 1.0
    
    def update(self, current_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        """
        Update beliefs based on sensor data from current location.
        
        Args:
            current_loc: (row, col) position where sensor reading was taken
            sensor_data: [(heard_white, felt_white), (heard_black, felt_black)]
        """
        # Store sensor reading
        self.sensor_readings[current_loc] = sensor_data
        
        # Update beliefs using Bayesian inference
        obs_white = sensor_data[0]  # (heard, felt) for white trapdoor
        obs_black = sensor_data[1]  # (heard, felt) for black trapdoor
        
        self._bayes_update(self.belief_white, current_loc, obs_white)
        self._bayes_update(self.belief_black, current_loc, obs_black)
    
    def _bayes_update(self, belief_grid: np.ndarray, current_loc: Tuple[int, int], observation: Tuple[bool, bool]):
        """
        Update belief grid using Bayesian inference.
        
        Args:
            belief_grid: Belief grid for one trapdoor color (white or black)
            current_loc: Location where observation was made
            observation: (heard, felt) tuple
        """
        heard, felt = observation
        row, col = current_loc
        
        # Calculate likelihood for each possible trapdoor location
        likelihood_mask = np.ones_like(belief_grid)
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                if belief_grid[r, c] == 0:
                    continue  # Skip impossible locations
                
                # Calculate distance
                dx = abs(r - row)
                dy = abs(c - col)
                
                # Get probabilities of hearing/feeling if trapdoor were here
                p_h = prob_hear(dx, dy)
                p_f = prob_feel(dx, dy)
                
                # Calculate likelihood of this observation
                # P(obs | trapdoor at (r,c)) = P(heard | trapdoor) * P(felt | trapdoor)
                prob_obs = 1.0
                prob_obs *= p_h if heard else (1.0 - p_h)
                prob_obs *= p_f if felt else (1.0 - p_f)
                
                likelihood_mask[r, c] = prob_obs
        
        # Bayesian update: P(trapdoor | obs) ∝ P(obs | trapdoor) * P(trapdoor)
        belief_grid *= likelihood_mask
        
        # Normalize
        total = np.sum(belief_grid)
        if total > 0:
            belief_grid /= total
        else:
            # If all probabilities are zero, reinitialize
            self._initialize_priors()
    
    def get_risk(self, loc: Tuple[int, int]) -> float:
        """
        Get the probability that a trapdoor is at this location.
        Returns value between 0.0 and 1.0.
        """
        if loc in self.known_trapdoors:
            return 1.0
        
        r, c = loc
        if (r + c) % 2 == 0:  # White square
            return float(self.belief_white[r, c])
        else:  # Black square
            return float(self.belief_black[r, c])
    
    def get_high_risk_squares(self, threshold: float = 0.2) -> Set[Tuple[int, int]]:
        """Get all squares with risk above threshold."""
        high_risk = set()
        for r in range(self.map_size):
            for c in range(self.map_size):
                if self.get_risk((r, c)) > threshold:
                    high_risk.add((r, c))
        return high_risk


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    Each node represents a game state.
    """
    
    def __init__(self, board_state: board.Board, parent=None, move=None):
        self.board_state = board_state
        self.parent = parent
        self.move = move  # The move that led to this state
        self.children = []
        self.visits = 0
        self.wins = 0.0  # Accumulated value (can be fractional for non-terminal states)
        self.untried_moves = board_state.get_valid_moves()
        self.is_terminal = board_state.is_game_over()
        
    def is_fully_expanded(self):
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration_constant=1.414):
        """
        Select the best child using UCB1 formula.
        UCB1 = (wins / visits) + c * sqrt(ln(parent_visits) / visits)
        """
        if not self.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                return child  # Return unvisited child immediately
            
            exploitation = child.wins / child.visits
            exploration = exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits
            )
            ucb1_score = exploitation + exploration
            
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child
        
        return best_child
    
    def expand(self, filter_turds: bool = False, enemy_pos: Tuple[int, int] = None, player_pos: Tuple[int, int] = None):
        """
        Expand the tree by adding a new child node.
        
        Args:
            filter_turds: If True, filter out turd moves unless enemy is adjacent
            enemy_pos: Enemy position for turd filtering
            player_pos: Player position for turd filtering
        """
        if not self.untried_moves:
            return None
        
        # Filter turds if requested - but don't modify untried_moves yet
        moves_to_try = list(self.untried_moves)
        if filter_turds and enemy_pos is not None and player_pos is not None:
            dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
            if dist_to_enemy != 1:
                # Remove turd moves if enemy is not adjacent
                moves_to_try = [m for m in moves_to_try if m[1] != MoveType.TURD]
        
        if not moves_to_try:
            # Mark all turd moves as tried if we're filtering them
            if filter_turds and enemy_pos is not None and player_pos is not None:
                dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
                if dist_to_enemy != 1:
                    # Remove turd moves from untried_moves
                    self.untried_moves = [m for m in self.untried_moves if m[1] != MoveType.TURD]
            return None
        
        # Take first move from filtered list
        move = moves_to_try[0]
        # Remove from untried_moves (use pop with index to maintain original behavior)
        if move in self.untried_moves:
            idx = self.untried_moves.index(move)
            self.untried_moves.pop(idx)
        
        dir, move_type = move
        
        # Create new board state with this move
        new_board = self.board_state.forecast_move(dir, move_type, check_ok=False)
        if new_board is None:
            return None
        
        # After forecast_move, turn switches to opponent
        # Reverse perspective so we're always evaluating from "player" perspective
        new_board.reverse_perspective()
        
        child = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child)
        return child
    
    def update(self, value: float):
        """
        Backpropagate the simulation result up the tree.
        Value should be from root player's perspective.
        """
        self.visits += 1
        self.wins += value
        
        if self.parent:
            # Value is from root player's perspective
            # Parent is opponent's move, so flip the value
            self.parent.update(-value)


class PlayerAgent:
    """
    Monte Carlo Tree Search agent for the chicken game.
    Uses MCTS with improved trapdoor avoidance and egg-laying strategy.
    """
    
    def __init__(self, board: board.Board, time_left: Callable):
        self.time_left = time_left
        self.my_parity = board.chicken_player.even_chicken
        
        # Trapdoor tracking using Bayesian inference
        self.oracle = TrapdoorOracle(map_size=board.game_map.MAP_SIZE)
        
        # Track visited positions
        self.visited_positions = set()
        self.visited_positions.add(board.chicken_player.get_location())
        
        # Track egg-laying squares
        self.visited_egg_squares = set()
        self.all_egg_squares = self._find_all_egg_squares(board)
        
        # Track known trapdoors (we stepped on them)
        self.known_trapdoors = set()
        
        # Track last move destination to detect trapdoor hits
        self.last_move_destination = None
        self.start_pos = board.chicken_player.get_spawn()
        
        # Track recent positions to avoid bouncing
        self.recent_positions = []  # Keep last 5 positions
        self.max_recent_positions = 5
        
        # Track squares where we've already laid eggs (to avoid revisiting)
        self.eggs_laid_positions = set()
        
        # Track if we've reached a corner (for early game strategy)
        self.reached_corner = False
        
        # MCTS parameters
        self.max_simulations = 1000
        self.simulation_depth = 20
        self.exploration_constant = 1.414
    
    def _find_all_egg_squares(self, board_state: board.Board) -> set:
        """Find all squares where we can potentially lay eggs."""
        egg_squares = set()
        even_chicken = board_state.chicken_player.even_chicken
        
        for x in range(8):
            for y in range(8):
                pos = (x, y)
                if (x + y) % 2 == even_chicken:
                    egg_squares.add(pos)
        
        return egg_squares
    
    def _get_position_after_move(self, current_pos: Tuple[int, int], direction: Direction) -> Tuple[int, int]:
        """Helper to calculate position after a move."""
        return loc_after_direction(current_pos, direction)
    
    def _can_lay_egg_at_position(self, pos: Tuple[int, int], board_state: board.Board) -> bool:
        """Check if we can lay an egg at a given position."""
        if (pos[0] + pos[1]) % 2 != board_state.chicken_player.even_chicken:
            return False
        if pos in board_state.eggs_player or pos in board_state.eggs_enemy:
            return False
        if pos in board_state.turds_player or pos in board_state.turds_enemy:
            return False
        if board_state.is_cell_in_enemy_turd_zone(pos):
            return False
        return True
    
    def _get_unvisited_egg_squares(self, board_state: board.Board) -> Set[Tuple[int, int]]:
        """Get egg-laying squares we haven't visited yet."""
        unvisited = set()
        for pos in self.all_egg_squares:
            if self._can_lay_egg_at_position(pos, board_state):
                if pos not in self.visited_egg_squares:
                    unvisited.add(pos)
        return unvisited
    
    def _distance_to_nearest_unvisited_egg_square(self, pos: Tuple[int, int], board_state: board.Board) -> int:
        """Calculate distance to nearest unvisited egg-laying square."""
        unvisited = self._get_unvisited_egg_squares(board_state)
        if not unvisited:
            return 0
        return min(abs(pos[0] - e[0]) + abs(pos[1] - e[1]) for e in unvisited)
    
    def _find_closest_unvisited_corner(self, pos: Tuple[int, int], board_state: board.Board) -> Optional[Tuple[int, int]]:
        """
        Find the closest unvisited corner where we can lay an egg.
        Only considers corners that match our parity (where we can actually lay eggs).
        """
        unvisited = self._get_unvisited_egg_squares(board_state)
        if not unvisited:
            return None
        
        # All 4 corners
        all_corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        
        # Filter to only corners where we can lay eggs based on our parity
        # Player A (even_chicken=0): can lay on (row+col) % 2 == 0
        # Player B (even_chicken=1): can lay on (row+col) % 2 == 1
        even_chicken = board_state.chicken_player.even_chicken
        valid_corners = [c for c in all_corners if (c[0] + c[1]) % 2 == even_chicken]
        
        # Further filter to unvisited corners
        unvisited_corners = [c for c in valid_corners if c in unvisited]
        
        if not unvisited_corners:
            return None
        
        # Return closest corner
        closest = min(unvisited_corners, key=lambda c: abs(pos[0] - c[0]) + abs(pos[1] - c[1]))
        return closest
    
    def _find_closest_valid_corner(self, pos: Tuple[int, int], board_state: board.Board) -> Optional[Tuple[int, int]]:
        """
        Find the closest corner where we can lay an egg (even if already visited).
        Used for early game strategy to get to ANY corner first.
        """
        # All 4 corners
        all_corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        
        # Filter to only corners where we can lay eggs based on our parity
        even_chicken = board_state.chicken_player.even_chicken
        valid_corners = [c for c in all_corners if (c[0] + c[1]) % 2 == even_chicken]
        
        if not valid_corners:
            return None
        
        # Return closest corner (even if we've already laid an egg there)
        closest = min(valid_corners, key=lambda c: abs(pos[0] - c[0]) + abs(pos[1] - c[1]))
        return closest
    
    def _simple_path_to_corner(self, current_pos: Tuple[int, int], target_corner: Tuple[int, int], board_state: board.Board, valid_moves: List[Tuple[Direction, MoveType]]) -> Optional[Tuple[Direction, MoveType]]:
        """
        Simple greedy pathfinding to get to a corner.
        Returns the best move to get closer to the target corner, avoiding trapdoors.
        Prioritizes laying eggs on valid egg-laying squares along the way.
        """
        if not valid_moves:
            return None
        
        # Calculate distance to corner
        dist_to_corner = abs(current_pos[0] - target_corner[0]) + abs(current_pos[1] - target_corner[1])
        
        # Check if we can lay an egg at current position
        can_lay_egg_here = board_state.can_lay_egg() and current_pos not in board_state.eggs_player
        even_chicken = board_state.chicken_player.even_chicken
        current_parity = (current_pos[0] + current_pos[1]) % 2
        is_valid_egg_square = (current_parity == even_chicken)
        
        # Filter out moves that step on known/high-risk trapdoors
        safe_moves = []
        for move in valid_moves:
            dir, move_type = move
            new_pos = self._get_position_after_move(current_pos, dir)
            
            # Never step on known trapdoors
            if new_pos in self.known_trapdoors or new_pos in board_state.found_trapdoors:
                continue
            
            # Avoid high-risk trapdoors
            trapdoor_risk = self.oracle.get_risk(new_pos)
            if trapdoor_risk > 0.2:  # Aggressive avoidance
                continue
            
            # Skip blocked positions
            if board_state.is_cell_blocked(new_pos):
                continue
            
            safe_moves.append(move)
        
        if not safe_moves:
            # If all moves are risky, use least risky ones
            def trapdoor_risk_score(move):
                dir, move_type = move
                new_pos = self._get_position_after_move(current_pos, dir)
                return self.oracle.get_risk(new_pos)
            
            safe_moves = sorted(valid_moves, key=trapdoor_risk_score)[:5]  # Top 5 safest
        
        # PRIORITY 1: If we can lay an egg here and we're not very close to corner, do it!
        # CRITICAL: Only select egg moves that move us TOWARD the corner!
        if can_lay_egg_here and is_valid_egg_square:
            egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
            if egg_moves:
                # Filter to only egg moves that move us TOWARD the corner
                egg_moves_toward_corner = []
                for move in egg_moves:
                    dir, move_type = move
                    new_pos = self._get_position_after_move(current_pos, dir)
                    dist_after = abs(new_pos[0] - target_corner[0]) + abs(new_pos[1] - target_corner[1])
                    if dist_after <= dist_to_corner:  # Moving toward or same distance (not away)
                        egg_moves_toward_corner.append(move)
                
                if egg_moves_toward_corner:
                    # Use the move that gets us closest to corner
                    best_egg_move = min(egg_moves_toward_corner,
                                       key=lambda m: abs(self._get_position_after_move(current_pos, m[0])[0] - target_corner[0]) + 
                                                     abs(self._get_position_after_move(current_pos, m[0])[1] - target_corner[1]))
                    return best_egg_move
                # If no moves toward corner, don't lay egg here - continue moving instead
        
        # PRIORITY 2: If we're at the corner, lay an egg!
        # When at corner, any direction is fine (we're already there)
        if current_pos == target_corner:
            if can_lay_egg_here and is_valid_egg_square:
                egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
                if egg_moves:
                    # At corner, prefer staying at corner (any direction that keeps us close)
                    # But if we must move, prefer moves that don't take us too far
                    best_egg_move = None
                    best_dist_after = float('inf')
                    for move in egg_moves:
                        dir, move_type = move
                        new_pos = self._get_position_after_move(current_pos, dir)
                        dist_after = abs(new_pos[0] - target_corner[0]) + abs(new_pos[1] - target_corner[1])
                        if dist_after < best_dist_after:
                            best_dist_after = dist_after
                            best_egg_move = move
                    if best_egg_move:
                        return best_egg_move
                    return egg_moves[0]  # Fallback
            # If we can't lay egg but can move, prefer plain move to stay at corner
            plain_moves = [m for m in safe_moves if m[1] == MoveType.PLAIN]
            if plain_moves:
                # Prefer moves that keep us at or near corner
                best_plain_move = None
                best_dist_after = float('inf')
                for move in plain_moves:
                    dir, move_type = move
                    new_pos = self._get_position_after_move(current_pos, dir)
                    dist_after = abs(new_pos[0] - target_corner[0]) + abs(new_pos[1] - target_corner[1])
                    if dist_after < best_dist_after:
                        best_dist_after = dist_after
                        best_plain_move = move
                if best_plain_move:
                    return best_plain_move
                return plain_moves[0]  # Fallback
        
        # PRIORITY 3: Move toward the corner
        # Score moves: prefer moves that get us closer, but also consider egg laying opportunities
        # CRITICAL: Penalize egg moves that move us AWAY from the corner!
        best_move = None
        best_score = float('-inf')
        
        for move in safe_moves:
            dir, move_type = move
            new_pos = self._get_position_after_move(current_pos, dir)
            dist_after = abs(new_pos[0] - target_corner[0]) + abs(new_pos[1] - target_corner[1])
            
            score = 0
            
            # If this move gets us to the corner, it's very good
            if new_pos == target_corner:
                score += 10000
                # If we can lay an egg at the corner, even better
                new_pos_parity = (new_pos[0] + new_pos[1]) % 2
                if new_pos_parity == even_chicken and move_type == MoveType.EGG:
                    score += 50000  # Perfect: at corner and laying egg
            else:
                # Score based on how much closer we get
                dist_improvement = dist_to_corner - dist_after
                score += dist_improvement * 1000
                
                # CRITICAL: If this is an EGG move that moves us AWAY from corner, heavily penalize it
                if move_type == MoveType.EGG and dist_after > dist_to_corner:
                    score -= 50000  # Massive penalty for egg moves that move away from corner
                
                # Bonus if this is a valid egg-laying square (we can lay egg next turn)
                new_pos_parity = (new_pos[0] + new_pos[1]) % 2
                if new_pos_parity == even_chicken:
                    score += 100  # Bonus for moving to egg-laying square
            
            # Prefer PLAIN moves over TURD moves (unless we're at corner)
            if move_type == MoveType.PLAIN:
                score += 10
            elif move_type == MoveType.TURD:
                score -= 1000  # Avoid turds when going to corner
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _find_path_to_unvisited_egg_square(self, start_pos: Tuple[int, int], board_state: board.Board, max_depth: int = 5) -> Optional[Tuple[int, int]]:
        """Find the best next position to move toward unvisited egg squares using BFS, avoiding trapdoors and egg squares."""
        unvisited = self._get_unvisited_egg_squares(board_state)
        if not unvisited:
            return None
        
        from collections import deque
        
        queue = deque([(start_pos, [])])
        visited_bfs = {start_pos}
        
        while queue and len(queue[0][1]) < max_depth:
            current, path = queue.popleft()
            
            # Check all 4 directions
            for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                next_pos = self._get_position_after_move(current, direction)
                
                # Skip invalid positions
                if not board_state.is_valid_cell(next_pos):
                    continue
                
                # Skip trapdoors (known and high-risk)
                if (next_pos in self.known_trapdoors or 
                    next_pos in board_state.found_trapdoors or
                    self.oracle.get_risk(next_pos) > 0.15):
                    continue
                
                # Skip blocked positions
                if board_state.is_cell_blocked(next_pos):
                    continue
                
                # CRITICAL: Skip squares where we've already laid eggs (unless it's the target)
                if next_pos in board_state.eggs_player and next_pos not in unvisited:
                    continue
                
                if next_pos in visited_bfs:
                    continue
                
                visited_bfs.add(next_pos)
                new_path = path + [next_pos]
                
                # If this is an unvisited egg square, return the first step
                if next_pos in unvisited:
                    return new_path[0] if new_path else next_pos
                
                queue.append((next_pos, new_path))
        
        # If no path found, return closest unvisited egg square direction
        closest = min(unvisited, key=lambda e: abs(start_pos[0] - e[0]) + abs(start_pos[1] - e[1]))
        dx = closest[0] - start_pos[0]
        dy = closest[1] - start_pos[1]
        
        if abs(dx) > abs(dy):
            return (start_pos[0] + (1 if dx > 0 else -1), start_pos[1])
        else:
            return (start_pos[0], start_pos[1] + (1 if dy > 0 else -1))
    
    def _evaluate_state(self, board_state: board.Board) -> float:
        """
        Evaluate a board state from the current player's perspective.
        Returns a value between -1 and 1, where 1 is best for player.
        """
        if board_state.is_game_over():
            return self._evaluate_terminal(board_state)
        
        # Primary factor: egg count difference
        player_eggs = board_state.chicken_player.get_eggs_laid()
        enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        egg_diff = player_eggs - enemy_eggs
        
        # Normalize egg difference
        normalized_egg_diff = egg_diff / 50.0
        score = normalized_egg_diff * 0.8  # 80% weight on eggs
        
        # Corner eggs bonus - corners are worth 3 points!
        corner_bonus = 0
        for egg_pos in board_state.eggs_player:
            if (egg_pos[0] == 0 or egg_pos[0] == 7) and (egg_pos[1] == 0 or egg_pos[1] == 7):
                corner_bonus += 2  # Already counted as 1, so add 2 more for total of 3
        score += (corner_bonus / 50.0) * 0.2  # Higher weight for corners
        
        # Positional evaluation - prioritize unvisited egg squares
        player_pos = board_state.chicken_player.get_location()
        
        # MAJOR PENALTY for being on a square where we've already laid an egg (unless necessary to reach new areas)
        if player_pos in board_state.eggs_player:
            # Check if there are unvisited egg squares nearby - if yes, this is wasteful
            unvisited = self._get_unvisited_egg_squares(board_state)
            if unvisited:
                dist_to_unvisited = self._distance_to_nearest_unvisited_egg_square(player_pos, board_state)
                if dist_to_unvisited <= 3:
                    score -= 0.4  # Large penalty for being on egg square when unvisited ones are nearby
                else:
                    score -= 0.1  # Smaller penalty if unvisited squares are far (might be necessary to traverse)
        
        # Check if we can lay an egg here
        can_lay_egg_here = board_state.can_lay_egg()
        if can_lay_egg_here and player_pos not in board_state.eggs_player:
            # Check if it's a corner (worth 3 points!)
            is_corner = (player_pos[0] == 0 or player_pos[0] == 7) and (player_pos[1] == 0 or player_pos[1] == 7)
            if is_corner:
                score += 0.4  # Huge bonus for corner egg squares
            else:
                # Check if it's an unvisited egg square
                if player_pos in self._get_unvisited_egg_squares(board_state):
                    score += 0.25  # Big bonus for unvisited egg squares
                else:
                    score += 0.15  # Bonus for being on egg-laying square
        
        # Bonus for being near unvisited egg squares
        dist_to_unvisited = self._distance_to_nearest_unvisited_egg_square(player_pos, board_state)
        if dist_to_unvisited > 0:
            score += max(0, 0.15 - dist_to_unvisited * 0.03)  # Closer is better
        
        # Bonus for being on a path to unvisited egg squares
        target_pos = self._find_path_to_unvisited_egg_square(player_pos, board_state, max_depth=3)
        if target_pos is not None:
            score += 0.1  # On path to egg square
        
        # Trapdoor avoidance - CRITICAL
        trapdoor_risk = self.oracle.get_risk(player_pos)
        if trapdoor_risk > 0.3:
            score -= 0.5  # Massive penalty for high-risk squares
        elif trapdoor_risk > 0.15:
            score -= 0.2  # Large penalty for medium-risk squares
        elif trapdoor_risk > 0.05:
            score -= 0.05  # Small penalty for low-risk squares
        
        # Mobility bonus
        try:
            player_moves = len(board_state.get_valid_moves())
            enemy_moves = len(board_state.get_valid_moves(enemy=True))
            
            if enemy_moves == 0 and board_state.turns_left_enemy > 0:
                score += 0.3  # Opponent blocked
            
            mobility_diff = (player_moves - enemy_moves) / 20.0
            score += mobility_diff * 0.1
        except:
            pass
        
        # Turd strategy - ONLY valuable when enemy is bordering (1 square away)
        enemy_pos = board_state.chicken_enemy.get_location()
        player_pos = board_state.chicken_player.get_location()
        dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
        
        # Penalize turds if enemy is NOT adjacent
        for turd_pos in board_state.turds_player:
            turd_dist_to_enemy = abs(turd_pos[0] - enemy_pos[0]) + abs(turd_pos[1] - enemy_pos[1])
            if turd_dist_to_enemy == 1:
                score += 0.1  # Turd blocking adjacent enemy
            else:
                score -= 0.2  # Penalty for wasteful turd placement
        
        # Clamp score to [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _evaluate_terminal(self, board_state: board.Board) -> float:
        """Evaluate a terminal state."""
        player_eggs = board_state.chicken_player.get_eggs_laid()
        enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        
        if player_eggs > enemy_eggs:
            return 1.0
        elif player_eggs < enemy_eggs:
            return -1.0
        else:
            return 0.0
    
    def _simulate(self, board_state: board.Board, is_root_player_turn: bool = True) -> float:
        """
        Simulate a random game from the given state.
        Returns the final evaluation from root player's perspective.
        """
        current_state = board_state.get_copy()
        depth = 0
        current_is_root_turn = is_root_player_turn
        
        while not current_state.is_game_over() and depth < self.simulation_depth:
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            # Use heuristic to bias random selection (light playouts)
            move = self._select_move_heuristic(current_state, valid_moves)
            
            dir, move_type = move
            new_state = current_state.forecast_move(dir, move_type, check_ok=False)
            if new_state is None:
                move = random.choice(valid_moves)
                dir, move_type = move
                new_state = current_state.forecast_move(dir, move_type, check_ok=False)
                if new_state is None:
                    break
            
            new_state.reverse_perspective()
            current_state = new_state
            current_is_root_turn = not current_is_root_turn
            depth += 1
        
        # Evaluate final state
        value = self._evaluate_state(current_state)
        
        # Convert to root player's perspective
        if not current_is_root_turn:
            value = -value
        
        return value
    
    def _select_move_heuristic(self, board_state: board.Board, valid_moves: List[Tuple[Direction, MoveType]]) -> Tuple[Direction, MoveType]:
        """
        Select a move using heuristic (for light playouts).
        Prioritizes egg laying and avoids trapdoors.
        """
        # Filter out moves that step on known/high-risk trapdoors
        safe_moves = []
        current_pos = board_state.chicken_player.get_location()
        
        for move in valid_moves:
            dir, move_type = move
            new_pos = self._get_position_after_move(current_pos, dir)
            
            # Never step on known trapdoors
            if new_pos in self.known_trapdoors or new_pos in board_state.found_trapdoors:
                continue
            
            # Avoid high-risk trapdoors
            trapdoor_risk = self.oracle.get_risk(new_pos)
            if trapdoor_risk > 0.25:  # Aggressive avoidance
                continue
            
            safe_moves.append(move)
        
        if not safe_moves:
            safe_moves = valid_moves  # Fallback if all moves are risky
        
        # Prioritize egg moves
        egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
        if egg_moves and board_state.can_lay_egg():
            return random.choice(egg_moves)
        
        # Use weighted random selection
        weights = []
        for move in safe_moves:
            dir, move_type = move
            new_pos = self._get_position_after_move(current_pos, dir)
            weight = 1.0
            
            if move_type == MoveType.EGG:
                # Check if it's a corner (worth 3 points!)
                # Note: egg is laid at current_pos, not new_pos
                is_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
                # Check if this is an unvisited egg square
                is_unvisited = current_pos not in board_state.eggs_player
                if is_corner and is_unvisited:
                    weight = 30.0  # Corner eggs on unvisited squares are worth 3 points!
                elif is_unvisited:
                    weight = 15.0  # Unvisited egg squares
                elif is_corner:
                    weight = 5.0  # Corner but already visited
                else:
                    weight = 2.0  # Already visited egg square (low priority)
            elif move_type == MoveType.TURD:
                enemy_pos = board_state.chicken_enemy.get_location()
                dist = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
                if dist == 1:  # ONLY when enemy is bordering (adjacent)
                    weight = 3.0
                else:
                    weight = 0.0  # NEVER lay turds when enemy is not adjacent
            elif move_type == MoveType.PLAIN:
                # ABSOLUTE PENALTY for moving to a square where we've already laid an egg
                if new_pos in board_state.eggs_player:
                    # This should be filtered out, but if it somehow gets here, zero weight
                    weight = 0.0  # Should never happen
                else:
                    # Check if this is an unvisited egg square
                    unvisited = self._get_unvisited_egg_squares(board_state)
                    if new_pos in unvisited:
                        is_corner = (new_pos[0] == 0 or new_pos[0] == 7) and (new_pos[1] == 0 or new_pos[1] == 7)
                        if is_corner:
                            weight = 20.0  # Corner unvisited egg square - highest priority
                        else:
                            weight = 10.0  # Unvisited egg square
                    else:
                        # Check if we're moving toward closest corner
                        closest_corner = self._find_closest_unvisited_corner(current_pos, board_state)
                        if closest_corner:
                            dist_to_corner = abs(new_pos[0] - closest_corner[0]) + abs(new_pos[1] - closest_corner[1])
                            dist_before = abs(current_pos[0] - closest_corner[0]) + abs(current_pos[1] - closest_corner[1])
                            if dist_to_corner < dist_before:
                                weight = 8.0  # Moving toward corner
                            else:
                                weight = 2.0  # Not moving toward corner
                        else:
                            # Regular move - weight by distance to unvisited squares
                            dist = self._distance_to_nearest_unvisited_egg_square(new_pos, board_state)
                            dist_before = self._distance_to_nearest_unvisited_egg_square(current_pos, board_state)
                            if dist < dist_before:
                                weight = 5.0  # Moving closer to unvisited squares
                            else:
                                weight = 1.0  # Exploring but not necessarily closer
            
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(safe_moves)
        
        weights = [w / total_weight for w in weights]
        
        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return safe_moves[i]
        
        return safe_moves[-1]
    
    def _mcts_search(self, root_board: board.Board, time_budget: float, allowed_moves: List[Tuple[Direction, MoveType]] = None) -> Tuple[Direction, MoveType]:
        """
        Perform MCTS search to find the best move.
        
        Args:
            root_board: Current board state
            time_budget: Time budget for search
            allowed_moves: List of moves to consider (if None, uses all valid moves)
        """
        # Create root node with filtered moves
        if allowed_moves is not None:
            # Create a custom node with filtered moves
            root = MCTSNode(root_board)
            root.untried_moves = allowed_moves.copy()
        else:
            root = MCTSNode(root_board)
        
        simulations = 0
        start_time = self.time_left()
        
        # Adjust number of simulations based on time
        max_sims = min(self.max_simulations, int(time_budget * 100))
        
        while simulations < max_sims:
            # Check time
            elapsed = start_time - self.time_left()
            if elapsed >= time_budget * 0.9:
                break
            
            # Selection: traverse tree to leaf
            node = root
            depth = 0
            while not node.is_terminal and node.is_fully_expanded() and node.children:
                child = node.best_child(self.exploration_constant)
                if child is None:
                    break
                node = child
                depth += 1
                if depth > 10:
                    break
            
            # Expansion: add new child if possible
            if not node.is_terminal and not node.is_fully_expanded():
                # Get positions for turd filtering
                enemy_pos = node.board_state.chicken_enemy.get_location()
                player_pos = node.board_state.chicken_player.get_location()
                
                expanded_node = node.expand(filter_turds=True, enemy_pos=enemy_pos, player_pos=player_pos)
                if expanded_node is not None:
                    node = expanded_node
            
            # Simulation
            is_root_player_turn = (depth % 2 == 0)
            
            if node.is_terminal:
                value = self._evaluate_terminal(node.board_state)
                if not is_root_player_turn:
                    value = -value
            else:
                value = self._simulate(node.board_state, is_root_player_turn)
            
            # Backpropagation
            if node:
                node.update(value)
            
            simulations += 1
        
        # Select best move (most visited child, but prioritize egg moves and corners)
        if not root.children:
            if allowed_moves:
                return allowed_moves[0]
            valid_moves = root_board.get_valid_moves()
            return valid_moves[0] if valid_moves else (Direction.UP, MoveType.PLAIN)
        
        # Prioritize egg moves and corner egg moves, avoid revisiting squares with eggs
        def child_score(child):
            if child.move is None:
                return -1
            dir, move_type = child.move
            current_pos = root_board.chicken_player.get_location()
            new_pos = self._get_position_after_move(current_pos, dir)
            unvisited = self._get_unvisited_egg_squares(root_board)
            closest_corner = self._find_closest_unvisited_corner(current_pos, root_board) if unvisited else None
            
            # ABSOLUTE PENALTY for moving to a square where we've already laid an egg
            if new_pos in root_board.eggs_player:
                # This should never happen if filtering works, but massive penalty just in case
                return child.visits - 10000  # Should never happen
            
            # Egg moves are best
            if move_type == MoveType.EGG:
                # Check if it's a corner (egg is laid at current_pos)
                is_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
                is_unvisited = current_pos not in root_board.eggs_player
                if is_corner and is_unvisited:
                    return child.visits + 20000  # Corner eggs worth 3 points!
                elif is_unvisited:
                    return child.visits + 5000  # Unvisited egg square
                elif is_corner:
                    return child.visits + 1000  # Corner but already visited
                return child.visits + 100  # Already visited egg square (low priority)
            
            # Turds only if enemy is adjacent
            if move_type == MoveType.TURD:
                enemy_pos = root_board.chicken_enemy.get_location()
                dist = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
                if dist == 1:
                    return child.visits + 100  # Good turd placement
                return -1000  # Bad turd placement
            
            # Plain moves - prioritize corners first, then unvisited egg squares
            if closest_corner:
                dist_to_corner = abs(new_pos[0] - closest_corner[0]) + abs(new_pos[1] - closest_corner[1])
                dist_before = abs(current_pos[0] - closest_corner[0]) + abs(current_pos[1] - closest_corner[1])
                
                if new_pos == closest_corner:
                    return child.visits + 10000  # Reached the corner!
                elif dist_to_corner < dist_before:
                    return child.visits + 5000 + (10 - dist_to_corner) * 100  # Moving toward corner
                elif dist_to_corner == dist_before:
                    return child.visits + 2000  # Same distance to corner
                else:
                    return child.visits + 500  # Moving away from corner
            
            # Check if this is an unvisited egg square
            if new_pos in unvisited:
                is_corner = (new_pos[0] == 0 or new_pos[0] == 7) and (new_pos[1] == 0 or new_pos[1] == 7)
                if is_corner:
                    return child.visits + 8000  # Corner unvisited egg square
                return child.visits + 3000  # Unvisited egg square
            
            # Move closer to unvisited squares
            dist = self._distance_to_nearest_unvisited_egg_square(new_pos, root_board)
            dist_before = self._distance_to_nearest_unvisited_egg_square(current_pos, root_board)
            
            if dist < dist_before:
                return child.visits + 1500 + (10 - dist) * 50  # Moving closer
            elif dist == dist_before:
                return child.visits + 500  # Same distance
            else:
                return child.visits + 100  # Moving away but exploring
            
            return child.visits
        
        best_child = max(root.children, key=child_score)
        return best_child.move
    
    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        """
        Play a move using Monte Carlo Tree Search.
        """
        self.time_left = time_left
        current_pos = board.chicken_player.get_location()
        
        # ABSOLUTE PRIORITY #1: If we're on a corner and can lay an egg, do it IMMEDIATELY!
        # Check this FIRST before anything else - corners are worth 3 points!
        # Corners are: (0,0), (0,7), (7,0), (7,7)
        is_on_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
        if is_on_corner:
            even_chicken = board.chicken_player.even_chicken
            corner_parity = (current_pos[0] + current_pos[1]) % 2
            # Check if this corner matches our parity (we can lay eggs here)
            if corner_parity == even_chicken:
                # This is a valid corner for us! Check if we can lay an egg
                # can_lay_egg() checks: parity matches AND square is empty
                if board.can_lay_egg():
                    # Double-check we haven't already laid an egg here
                    if current_pos not in board.eggs_player:
                        # Get ALL valid moves (don't filter yet) - we need egg moves
                        all_valid_moves = board.get_valid_moves()
                        corner_egg_moves = [m for m in all_valid_moves if m[1] == MoveType.EGG]
                        if corner_egg_moves:
                            # LAY EGG ON CORNER IMMEDIATELY - highest priority!
                            # Corner eggs are worth 3 points!
                            self.last_move_destination = current_pos
                            self.reached_corner = True  # Mark that we've reached and laid egg at corner
                            return corner_egg_moves[0]
                        # If no egg moves found, something is wrong - but continue with normal logic
                    # Already has egg - continue
                # Can't lay egg here (maybe blocked) - continue
        
        # Check if we stepped on a trapdoor (got teleported back to start)
        if current_pos == self.start_pos and self.last_move_destination is not None:
            # We were teleported back - the last move destination was a trapdoor!
            if self.last_move_destination != self.start_pos:
                # Mark it as a known trapdoor
                self.known_trapdoors.add(self.last_move_destination)
                self.oracle.add_known_trapdoor(self.last_move_destination)
                self.last_move_destination = None  # Reset
        
        # Update trapdoor oracle with sensor data
        self.oracle.update(current_pos, sensor_data)
        
        # Update known trapdoors from board
        for trapdoor_pos in board.found_trapdoors:
            if trapdoor_pos not in self.known_trapdoors:
                self.known_trapdoors.add(trapdoor_pos)
                self.oracle.add_known_trapdoor(trapdoor_pos)
        
        # Track visited positions
        self.visited_positions.add(current_pos)
        
        # Track recent positions to avoid bouncing
        self.recent_positions.append(current_pos)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)
        
        # Track egg-laying squares
        if board.can_lay_egg() and current_pos not in board.eggs_player:
            self.visited_egg_squares.add(current_pos)
        self.visited_egg_squares.update(board.eggs_player)
        
        # Track squares where we've laid eggs
        self.eggs_laid_positions.update(board.eggs_player)
        
        # Check if we've reached a corner (for early game strategy)
        is_on_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
        if is_on_corner:
            even_chicken = board.chicken_player.even_chicken
            corner_parity = (current_pos[0] + current_pos[1]) % 2
            if corner_parity == even_chicken:
                # We're on a valid corner - mark that we've reached one
                self.reached_corner = True
        
        # EARLY GAME STRATEGY: If we haven't reached a corner yet, use simple pathfinding
        # Skip MCTS entirely until we get to a corner
        # IMPORTANT: Lay eggs on valid squares along the way to the corner!
        if not self.reached_corner:
            # Find the closest valid corner
            target_corner = self._find_closest_valid_corner(current_pos, board)
            
            if target_corner:
                # Calculate distance to corner
                dist_to_corner = abs(current_pos[0] - target_corner[0]) + abs(current_pos[1] - target_corner[1])
                
                # If we're already at the corner, try to lay an egg
                if current_pos == target_corner:
                    if board.can_lay_egg() and current_pos not in board.eggs_player:
                        valid_moves = board.get_valid_moves()
                        egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
                        if egg_moves:
                            self.last_move_destination = current_pos
                            self.reached_corner = True  # Mark as reached after laying egg
                            return egg_moves[0]
                
                # PRIORITY: If we can lay an egg here (and haven't already), do it!
                # But if we're very close to corner (1-2 moves away), prioritize getting there first
                # CRITICAL: When laying an egg, make sure the direction moves us TOWARD the corner!
                if board.can_lay_egg() and current_pos not in board.eggs_player:
                    # Check if we're on a valid egg-laying square
                    even_chicken = board.chicken_player.even_chicken
                    current_parity = (current_pos[0] + current_pos[1]) % 2
                    if current_parity == even_chicken:
                        # We're on a valid egg-laying square!
                        # If we're far from corner (3+ moves), lay egg here
                        # If we're close (1-2 moves), prioritize getting to corner first
                        if dist_to_corner >= 3:
                            # Far from corner - lay egg here, then continue toward corner
                            valid_moves = board.get_valid_moves()
                            egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
                            # Filter to only egg moves that move us TOWARD the corner
                            egg_moves_toward_corner = []
                            for move in egg_moves:
                                dir, move_type = move
                                new_pos = self._get_position_after_move(current_pos, dir)
                                dist_after = abs(new_pos[0] - target_corner[0]) + abs(new_pos[1] - target_corner[1])
                                if dist_after <= dist_to_corner:  # Moving toward or same distance
                                    egg_moves_toward_corner.append(move)
                            
                            if egg_moves_toward_corner:
                                # Use the move that gets us closest to corner
                                best_egg_move = min(egg_moves_toward_corner, 
                                                   key=lambda m: abs(self._get_position_after_move(current_pos, m[0])[0] - target_corner[0]) + 
                                                                 abs(self._get_position_after_move(current_pos, m[0])[1] - target_corner[1]))
                                dir, move_type = best_egg_move
                                self.last_move_destination = self._get_position_after_move(current_pos, dir)
                                return best_egg_move
                            elif egg_moves:
                                # If no moves toward corner, use any egg move (better than nothing)
                                dir, move_type = egg_moves[0]
                                self.last_move_destination = self._get_position_after_move(current_pos, dir)
                                return egg_moves[0]
                        elif dist_to_corner == 1:
                            # Very close to corner - get there first, then lay egg
                            # But if we can't move directly to corner, lay egg here (moving toward corner)
                            valid_moves = board.get_valid_moves()
                            moves_to_corner = []
                            for move in valid_moves:
                                dir, move_type = move
                                if move_type != MoveType.EGG:
                                    new_pos = self._get_position_after_move(current_pos, dir)
                                    if new_pos == target_corner:
                                        moves_to_corner.append(move)
                            
                            if moves_to_corner:
                                # Can reach corner in one move - go there first
                                dir, move_type = moves_to_corner[0]
                                self.last_move_destination = self._get_position_after_move(current_pos, dir)
                                return moves_to_corner[0]
                            else:
                                # Can't reach corner directly - lay egg here, but move toward corner
                                egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
                                if egg_moves:
                                    # Find egg move that moves us toward corner (should be all of them since we're 1 away)
                                    for move in egg_moves:
                                        dir, move_type = move
                                        new_pos = self._get_position_after_move(current_pos, dir)
                                        if new_pos == target_corner:
                                            self.last_move_destination = new_pos
                                            return move
                                    # If none move directly to corner, use first one
                                    dir, move_type = egg_moves[0]
                                    self.last_move_destination = self._get_position_after_move(current_pos, dir)
                                    return egg_moves[0]
                        else:
                            # 2 moves away - lay egg here, moving toward corner
                            valid_moves = board.get_valid_moves()
                            egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
                            if egg_moves:
                                # Filter to only egg moves that move us TOWARD the corner
                                egg_moves_toward_corner = []
                                for move in egg_moves:
                                    dir, move_type = move
                                    new_pos = self._get_position_after_move(current_pos, dir)
                                    dist_after = abs(new_pos[0] - target_corner[0]) + abs(new_pos[1] - target_corner[1])
                                    if dist_after < dist_to_corner:  # Must move closer
                                        egg_moves_toward_corner.append(move)
                                
                                if egg_moves_toward_corner:
                                    # Use the move that gets us closest to corner
                                    best_egg_move = min(egg_moves_toward_corner,
                                                       key=lambda m: abs(self._get_position_after_move(current_pos, m[0])[0] - target_corner[0]) + 
                                                                     abs(self._get_position_after_move(current_pos, m[0])[1] - target_corner[1]))
                                    dir, move_type = best_egg_move
                                    self.last_move_destination = self._get_position_after_move(current_pos, dir)
                                    return best_egg_move
                                else:
                                    # If no moves toward corner, use any egg move
                                    dir, move_type = egg_moves[0]
                                    self.last_move_destination = self._get_position_after_move(current_pos, dir)
                                    return egg_moves[0]
                
                # Use simple pathfinding to get to the corner (will prioritize egg laying along the way)
                valid_moves = board.get_valid_moves()
                best_move = self._simple_path_to_corner(current_pos, target_corner, board, valid_moves)
                
                if best_move:
                    dir, move_type = best_move
                    self.last_move_destination = self._get_position_after_move(current_pos, dir)
                    return best_move
                else:
                    # Fallback: if pathfinding fails, use greedy move toward corner
                    # Find the move that gets us closest to the corner
                    valid_moves = board.get_valid_moves()
                    if valid_moves:
                        best_move = None
                        best_dist = float('inf')
                        for move in valid_moves:
                            dir, move_type = move
                            new_pos = self._get_position_after_move(current_pos, dir)
                            # Skip known trapdoors
                            if new_pos in self.known_trapdoors or new_pos in board.found_trapdoors:
                                continue
                            dist = abs(new_pos[0] - target_corner[0]) + abs(new_pos[1] - target_corner[1])
                            if dist < best_dist:
                                best_dist = dist
                                best_move = move
                        if best_move:
                            dir, move_type = best_move
                            self.last_move_destination = self._get_position_after_move(current_pos, dir)
                            return best_move
        
        # Calculate time budget
        remaining_time = time_left()
        turns_left = board.turns_left_player
        
        if turns_left > 30:
            time_budget = min(remaining_time / (turns_left + 5), 5.0)
        elif turns_left > 15:
            time_budget = min(remaining_time / (turns_left + 3), 3.0)
        else:
            time_budget = min(remaining_time / (turns_left + 1), 1.0)
        
        time_budget = max(0.1, min(time_budget, remaining_time * 0.8))
        
        # ABSOLUTE PRIORITY: If we're on a corner and can lay an egg, do it IMMEDIATELY!
        # Check this BEFORE any filtering or other logic
        is_on_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
        if is_on_corner:
            # Check if this corner matches our parity (we can lay eggs here)
            even_chicken = board.chicken_player.even_chicken
            corner_parity = (current_pos[0] + current_pos[1]) % 2
            if corner_parity == even_chicken:
                # This is a valid corner for us to lay eggs!
                if board.can_lay_egg() and current_pos not in board.eggs_player:
                    # Get valid moves and find egg moves
                    valid_moves_temp = board.get_valid_moves()
                    corner_egg_moves = [m for m in valid_moves_temp if m[1] == MoveType.EGG]
                    if corner_egg_moves:
                        # Return immediately - lay egg on corner!
                        self.last_move_destination = current_pos
                        return corner_egg_moves[0]
        
        # Filter out moves that step on known/high-risk trapdoors - AGGRESSIVE AVOIDANCE
        valid_moves = board.get_valid_moves()
        safe_moves = []
        
        # CRITICAL: Filter out moves to squares where we've already laid eggs (unless absolutely necessary)
        unvisited = self._get_unvisited_egg_squares(board)
        has_unvisited_options = len(unvisited) > 0
        
        for move in valid_moves:
            dir, move_type = move
            new_pos = self._get_position_after_move(current_pos, dir)
            
            # Never step on known trapdoors
            if new_pos in self.known_trapdoors or new_pos in board.found_trapdoors:
                continue
            
            # Aggressively avoid high-risk trapdoors
            trapdoor_risk = self.oracle.get_risk(new_pos)
            if trapdoor_risk > 0.15:  # Lower threshold for more aggressive avoidance
                continue
            
            # CRITICAL: If there are unvisited egg squares available, NEVER move to a square with an egg
            if has_unvisited_options and new_pos in board.eggs_player:
                # Only allow if it's the ONLY way to reach unvisited squares (check if all other moves are blocked)
                # For now, completely filter out - we'll add back only if no other options
                continue
            
            safe_moves.append(move)
        
        # If we filtered out all moves because they go to egg squares, but we have no other options,
        # add back moves that go to egg squares (but only if they help reach unvisited squares)
        if not safe_moves and has_unvisited_options:
            for move in valid_moves:
                dir, move_type = move
                new_pos = self._get_position_after_move(current_pos, dir)
                
                # Skip trapdoors
                if new_pos in self.known_trapdoors or new_pos in board.found_trapdoors:
                    continue
                if self.oracle.get_risk(new_pos) > 0.15:
                    continue
                
                # Only add if moving here gets us closer to unvisited squares
                if new_pos in board.eggs_player:
                    dist_after = self._distance_to_nearest_unvisited_egg_square(new_pos, board)
                    dist_before = self._distance_to_nearest_unvisited_egg_square(current_pos, board)
                    if dist_after < dist_before:
                        safe_moves.append(move)
        
        # If no safe moves, use least risky ones
        if not safe_moves:
            def trapdoor_risk_score(move):
                dir, move_type = move
                new_pos = self._get_position_after_move(current_pos, dir)
                return self.oracle.get_risk(new_pos)
            
            valid_moves = sorted(valid_moves, key=trapdoor_risk_score)
            safe_moves = [m for m in valid_moves if trapdoor_risk_score(m) < 0.5]
            if not safe_moves:
                safe_moves = [valid_moves[0]]  # Last resort
        
        # STRATEGY: Prioritize corners first, then explore expansively
        unvisited = self._get_unvisited_egg_squares(board)
        closest_corner = self._find_closest_unvisited_corner(current_pos, board) if unvisited else None
        
        # STRATEGY: FIRST get to corner, laying eggs at every valid spot along the way
        
        # PRIORITY 1: If we're on a corner and can lay an egg, do it immediately!
        # (This is a redundant check after the early check above, but keep it for safety)
        # CRITICAL: This check MUST happen before MCTS or any other logic
        is_on_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
        if is_on_corner:
            even_chicken = board.chicken_player.even_chicken
            corner_parity = (current_pos[0] + current_pos[1]) % 2
            if corner_parity == even_chicken:
                # This corner matches our parity - we CAN lay eggs here
                if board.can_lay_egg() and current_pos not in board.eggs_player:
                    # We're on a valid corner - lay an egg immediately!
                    # Get egg moves from safe_moves (already filtered for trapdoors)
                    corner_egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
                    if corner_egg_moves:
                        self.last_move_destination = current_pos
                        return corner_egg_moves[0]  # Return immediately, don't use MCTS
                    # If no egg moves in safe_moves, try all valid moves
                    all_valid_moves = board.get_valid_moves()
                    corner_egg_moves_all = [m for m in all_valid_moves if m[1] == MoveType.EGG]
                    if corner_egg_moves_all:
                        # Even if trapdoor risk, corner is worth it (3 points!)
                        self.last_move_destination = current_pos
                        return corner_egg_moves_all[0]  # Return immediately!
        
        # PRIORITY 2: If we can lay an egg (on any valid egg-laying square), do it!
        # This ensures we lay eggs at every valid spot on the way to the corner
        if board.can_lay_egg() and current_pos not in board.eggs_player:
            egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
            if egg_moves:
                # If there's a corner to reach, check if we should prioritize moving toward it
                if closest_corner:
                    dist_to_corner_now = abs(current_pos[0] - closest_corner[0]) + abs(current_pos[1] - closest_corner[1])
                    
                    # If we're very close to corner (1 move away), prioritize getting there first
                    if dist_to_corner_now == 1:
                        # Get to corner first, then lay egg there
                        moves_to_corner = []
                        for move in safe_moves:
                            dir, move_type = move
                            if move_type != MoveType.EGG:
                                new_pos = self._get_position_after_move(current_pos, dir)
                                if new_pos == closest_corner:
                                    moves_to_corner.append(move)
                        
                        if moves_to_corner:
                            # Go to corner first
                            safe_moves = moves_to_corner
                        else:
                            # Can't reach corner in one move, lay egg here
                            safe_moves = egg_moves
                    else:
                        # We're not at corner yet - lay egg here, then continue toward corner
                        safe_moves = egg_moves
                else:
                    # No corner to reach - lay eggs expansively
                    safe_moves = egg_moves
        
        if safe_moves and unvisited:
            # Score moves by how close they get us to unvisited egg squares
            def move_score(move):
                dir, move_type = move
                if move_type == MoveType.EGG:
                    # Laying eggs is ALWAYS high priority - lay at every valid spot!
                    if current_pos in unvisited:
                        is_corner = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
                        if is_corner:
                            return 100000  # Corner egg on unvisited square is ABSOLUTE HIGHEST priority!
                        return 50000  # Egg on unvisited square - very high priority!
                    # Even if visited, laying eggs is still good (but lower priority)
                    return 10000  # Egg moves are always top priority
                
                new_pos = self._get_position_after_move(current_pos, dir)
                
                # ABSOLUTE PENALTY for moving to a square where we've already laid an egg
                # (This should already be filtered, but double-check)
                if new_pos in board.eggs_player:
                    return -10000  # Massive penalty - should never happen if filtering works
                
                # Check trapdoor risk
                risk = self.oracle.get_risk(new_pos)
                if risk > 0.1:
                    return -1000  # Avoid risky squares
                
                # Check if this position was recently visited (avoid bouncing)
                if new_pos in self.recent_positions:
                    return -1000  # Penalty for revisiting recent positions
                
                # PRIORITY 1: Move toward closest unvisited corner (HIGHEST PRIORITY)
                # BUT: If we can lay an egg, that's also highest priority (lay eggs along the way!)
                if closest_corner:
                    dist_to_corner = abs(new_pos[0] - closest_corner[0]) + abs(new_pos[1] - closest_corner[1])
                    dist_to_corner_before = abs(current_pos[0] - closest_corner[0]) + abs(current_pos[1] - closest_corner[1])
                    
                    if new_pos == closest_corner:
                        return 80000  # Reached the corner! Very high priority!
                    elif dist_to_corner < dist_to_corner_before:
                        # Moving closer to corner - prioritize based on how close
                        # BUT: If we can lay an egg at current_pos, that's EQUAL priority!
                        if move_type == MoveType.EGG and current_pos not in board.eggs_player:
                            return 75000  # Laying egg on way to corner - almost as high as reaching corner!
                        return 30000 + (20 - dist_to_corner) * 1000  # High priority for corner moves
                    elif dist_to_corner == dist_to_corner_before:
                        # Same distance - but if we can lay an egg, prioritize that MUCH more
                        if move_type == MoveType.EGG and current_pos not in board.eggs_player:
                            return 60000  # Laying egg is MUCH better than staying same distance
                        return 5000  # Same distance to corner
                    else:
                        return 100  # Moving away from corner (low priority)
                
                # PRIORITY 2: Check if this is an unvisited egg square
                if new_pos in unvisited:
                    is_corner = (new_pos[0] == 0 or new_pos[0] == 7) and (new_pos[1] == 0 or new_pos[1] == 7)
                    if is_corner:
                        return 6000  # Corner unvisited egg square
                    return 3000  # Unvisited egg square
                
                # PRIORITY 3: Move closer to nearest unvisited egg square
                dist = self._distance_to_nearest_unvisited_egg_square(new_pos, board)
                dist_before = self._distance_to_nearest_unvisited_egg_square(current_pos, board)
                
                if dist < dist_before:
                    return 1500 + (10 - dist)  # Moving closer to unvisited squares
                elif dist == dist_before:
                    return 500  # Same distance
                else:
                    return 100  # Moving away (but exploring new areas)
            
            safe_moves = sorted(safe_moves, key=move_score, reverse=True)
        
        # CRITICAL: Prioritize corner moves - but lay eggs along the way
        if closest_corner:
            dist_to_corner = abs(current_pos[0] - closest_corner[0]) + abs(current_pos[1] - closest_corner[1])
            
            # If we're at the corner and can lay an egg, do it immediately!
            if current_pos == closest_corner:
                even_chicken = board.chicken_player.even_chicken
                corner_parity = (current_pos[0] + current_pos[1]) % 2
                if corner_parity == even_chicken and board.can_lay_egg() and current_pos not in board.eggs_player:
                    corner_egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
                    if corner_egg_moves:
                        self.last_move_destination = current_pos
                        return corner_egg_moves[0]  # Return immediately!
            
            # If we're close to corner (1-2 moves), prioritize getting there
            # But if we can lay an egg here, do it first (unless we're 1 move from corner)
            if dist_to_corner <= 2:
                # If we're 1 move away, prioritize getting to corner first
                if dist_to_corner == 1:
                    moves_to_corner = []
                    for move in safe_moves:
                        dir, move_type = move
                        if move_type != MoveType.EGG:
                            new_pos = self._get_position_after_move(current_pos, dir)
                            if new_pos == closest_corner:
                                moves_to_corner.append(move)
                    
                    if moves_to_corner:
                        safe_moves = moves_to_corner + [m for m in safe_moves if m not in moves_to_corner]
                else:
                    # 2 moves away - prioritize moves that get us closer, but allow egg laying
                    moves_to_corner = []
                    for move in safe_moves:
                        dir, move_type = move
                        if move_type == MoveType.EGG:
                            # If we can lay an egg, prioritize it (we'll get to corner next turn)
                            moves_to_corner.append(move)
                        else:
                            new_pos = self._get_position_after_move(current_pos, dir)
                            if new_pos == closest_corner:
                                moves_to_corner.append(move)
                            elif abs(new_pos[0] - closest_corner[0]) + abs(new_pos[1] - closest_corner[1]) < dist_to_corner:
                                moves_to_corner.append(move)
                    
                    if moves_to_corner:
                        safe_moves = moves_to_corner + [m for m in safe_moves if m not in moves_to_corner]
        
        # CRITICAL: Filter turds - ONLY when enemy is bordering (1 square away, manhattan distance == 1)
        enemy_pos = board.chicken_enemy.get_location()
        dist_to_enemy = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
        
        # Remove ALL turd moves unless enemy is exactly 1 square away (bordering)
        if dist_to_enemy != 1:
            safe_moves = [m for m in safe_moves if m[1] != MoveType.TURD]
        
        # If we can lay an egg, ONLY consider egg moves (unless enemy is adjacent and we want to block)
        if board.can_lay_egg():
            egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
            if egg_moves:
                # Only use turds if enemy is adjacent AND we have egg moves available
                if dist_to_enemy == 1:
                    turd_moves = [m for m in safe_moves if m[1] == MoveType.TURD]
                    # Prefer eggs over turds even when enemy is adjacent
                    safe_moves = egg_moves + turd_moves
                else:
                    safe_moves = egg_moves  # Only eggs
        
        # If we have safe moves, prioritize corner directly if far away
        # BUT: Lay eggs at every valid spot along the way!
        if safe_moves and closest_corner:
            dist_to_corner = abs(current_pos[0] - closest_corner[0]) + abs(current_pos[1] - closest_corner[1])
            
            # If we're far from corner, use simple greedy move toward it
            # BUT if we can lay an egg here, do it first!
            if dist_to_corner > 3:
                # First check: Can we lay an egg here?
                if board.can_lay_egg() and current_pos not in board.eggs_player:
                    egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
                    if egg_moves:
                        # Lay egg here, then continue toward corner next turn
                        return egg_moves[0]
                
                # Can't lay egg here - move toward corner
                # Find the move that gets us closest to the corner
                best_move_to_corner = None
                best_dist = float('inf')
                
                for move in safe_moves:
                    dir, move_type = move
                    if move_type == MoveType.EGG:
                        continue  # Skip egg moves when far from corner (already handled above)
                    new_pos = self._get_position_after_move(current_pos, dir)
                    dist = abs(new_pos[0] - closest_corner[0]) + abs(new_pos[1] - closest_corner[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_move_to_corner = move
                
                if best_move_to_corner:
                    # Track move destination
                    dir, move_type = best_move_to_corner
                    self.last_move_destination = self._get_position_after_move(current_pos, dir)
                    return best_move_to_corner
        
        # If we have safe moves, use MCTS (but filter turds in MCTS too)
        if safe_moves:
            try:
                # Pass filtered moves to MCTS
                best_move = self._mcts_search(board, time_budget, allowed_moves=safe_moves)
                
                # Verify the move is safe
                dir, move_type = best_move
                new_pos = self._get_position_after_move(current_pos, dir)
                
                # Double-check safety
                if (new_pos in self.known_trapdoors or 
                    new_pos in board.found_trapdoors or 
                    self.oracle.get_risk(new_pos) > 0.2):
                    # Fallback to safest move
                    if safe_moves:
                        best_move = safe_moves[0]
                    else:
                        best_move = valid_moves[0]
                
                # FINAL SAFETY CHECK: Never return a turd move unless enemy is adjacent
                dir, move_type = best_move
                if move_type == MoveType.TURD:
                    enemy_pos = board.chicken_enemy.get_location()
                    dist_to_enemy = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
                    if dist_to_enemy != 1:
                        # Replace with best non-turd move
                        non_turd_moves = [m for m in safe_moves if m[1] != MoveType.TURD]
                        if non_turd_moves:
                            best_move = non_turd_moves[0]
                        else:
                            # Fallback to egg or plain move
                            egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
                            if egg_moves:
                                best_move = egg_moves[0]
                            else:
                                plain_moves = [m for m in safe_moves if m[1] == MoveType.PLAIN]
                                if plain_moves:
                                    best_move = plain_moves[0]
                
                # ABSOLUTE FINAL CHECK: If we're on a corner, we MUST lay an egg!
                # Override any other move if we're on a corner
                is_on_corner_final = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
                if is_on_corner_final:
                    even_chicken = board.chicken_player.even_chicken
                    corner_parity = (current_pos[0] + current_pos[1]) % 2
                    if corner_parity == even_chicken and board.can_lay_egg() and current_pos not in board.eggs_player:
                        dir, move_type = best_move
                        if move_type != MoveType.EGG:
                            # We're on a corner but not laying an egg - FIX THIS!
                            egg_moves_final = [m for m in safe_moves if m[1] == MoveType.EGG] if safe_moves else [m for m in valid_moves if m[1] == MoveType.EGG]
                            if egg_moves_final:
                                best_move = egg_moves_final[0]
                
                # Track move destination for trapdoor detection
                dir, move_type = best_move
                self.last_move_destination = self._get_position_after_move(current_pos, dir)
                
                return best_move
            except Exception as e:
                # Fallback to safest move
                if safe_moves:
                    best_move = safe_moves[0]
                else:
                    best_move = valid_moves[0] if valid_moves else (Direction.UP, MoveType.PLAIN)
                
                # FINAL SAFETY CHECK: Never return a turd move unless enemy is adjacent
                dir, move_type = best_move
                if move_type == MoveType.TURD:
                    enemy_pos = board.chicken_enemy.get_location()
                    dist_to_enemy = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
                    if dist_to_enemy != 1:
                        # Replace with best non-turd move
                        non_turd_moves = [m for m in safe_moves if m[1] != MoveType.TURD]
                        if non_turd_moves:
                            best_move = non_turd_moves[0]
                
                # ABSOLUTE FINAL CHECK: If we're on a corner, we MUST lay an egg!
                is_on_corner_final = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
                if is_on_corner_final:
                    even_chicken = board.chicken_player.even_chicken
                    corner_parity = (current_pos[0] + current_pos[1]) % 2
                    if corner_parity == even_chicken and board.can_lay_egg() and current_pos not in board.eggs_player:
                        dir, move_type = best_move
                        if move_type != MoveType.EGG:
                            egg_moves_final = [m for m in safe_moves if m[1] == MoveType.EGG] if safe_moves else [m for m in valid_moves if m[1] == MoveType.EGG]
                            if egg_moves_final:
                                best_move = egg_moves_final[0]
                
                # Track move destination
                dir, move_type = best_move
                self.last_move_destination = self._get_position_after_move(current_pos, dir)
                return best_move
        else:
            # No safe moves - use least risky
            best_move = safe_moves[0] if safe_moves else valid_moves[0]
            
            # ABSOLUTE FINAL CHECK: If we're on a corner, we MUST lay an egg!
            is_on_corner_final = (current_pos[0] == 0 or current_pos[0] == 7) and (current_pos[1] == 0 or current_pos[1] == 7)
            if is_on_corner_final:
                even_chicken = board.chicken_player.even_chicken
                corner_parity = (current_pos[0] + current_pos[1]) % 2
                if corner_parity == even_chicken and board.can_lay_egg() and current_pos not in board.eggs_player:
                    dir, move_type = best_move
                    if move_type != MoveType.EGG:
                        egg_moves_final = [m for m in safe_moves if m[1] == MoveType.EGG] if safe_moves else [m for m in valid_moves if m[1] == MoveType.EGG]
                        if egg_moves_final:
                            best_move = egg_moves_final[0]
            
            # FINAL SAFETY CHECK: Never return a turd move unless enemy is adjacent
            dir, move_type = best_move
            if move_type == MoveType.TURD:
                enemy_pos = board.chicken_enemy.get_location()
                dist_to_enemy = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
                if dist_to_enemy != 1:
                    # Replace with best non-turd move
                    non_turd_moves = [m for m in safe_moves if m[1] != MoveType.TURD] if safe_moves else [m for m in valid_moves if m[1] != MoveType.TURD]
                    if non_turd_moves:
                        best_move = non_turd_moves[0]
            
            # Track move destination
            dir, move_type = best_move
            self.last_move_destination = self._get_position_after_move(current_pos, dir)
            return best_move
