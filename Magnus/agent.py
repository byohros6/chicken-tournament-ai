"""
Magnus - The Strategic Chicken AI
Combines A* pathfinding, Bayesian trapdoor inference, and minimax search
"""
import time
import math
from typing import List, Tuple, Callable, Optional, Dict
from collections import defaultdict

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction

from .trapdoor_belief import TrapdoorBelief
from .pathfinding import PathFinder
from .strategy import StrategicPlanner


class PlayerAgent:
    """
    Magnus - A strategic chicken agent that combines:
    - Bayesian trapdoor belief tracking
    - A* pathfinding for optimal egg collection
    - Strategic turd placement
    - Minimax search with alpha-beta pruning
    - Iterative deepening
    """
    
    def __init__(self, board: Board, time_left: Callable):
        # Core components
        self.trapdoor_belief = TrapdoorBelief(board.game_map.MAP_SIZE)
        self.pathfinder = PathFinder(board)
        self.strategy = StrategicPlanner(board)
        
        # Game state tracking
        self.my_parity = board.chicken_player.even_chicken
        self.map_size = board.game_map.MAP_SIZE
        
        # Egg tour (precomputed path through all egg squares)
        self.egg_tour: List[Tuple[int, int]] = []
        self.tour_index = 0
        
        # Search parameters
        self.max_search_depth = 12
        self.time_safety_margin = 0.05  # Reserve 50ms for safety
        
        # Search enhancements
        self.transposition_table: Dict[int, Tuple[float, Optional[Tuple[Direction, MoveType]], int]] = {}
        self.killer_moves: Dict[int, List[Tuple[Direction, MoveType]]] = defaultdict(list)
        self.history_table: Dict[Tuple[Direction, MoveType], int] = defaultdict(int)
        
        # Performance tracking
        self.nodes_searched = 0
        self.cache_hits = 0
    
    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], 
             time_left: Callable) -> Tuple[Direction, MoveType]:
        """
        Main decision function called each turn.
        """
        start_time = time.time()
        self.nodes_searched = 0
        self.cache_hits = 0
        
        # Update trapdoor beliefs
        my_loc = board.chicken_player.get_location()
        self.trapdoor_belief.update(my_loc, sensor_data)
        
        # Check if we triggered a trapdoor (update beliefs)
        if hasattr(board, 'found_trapdoors'):
            for trap_loc in board.found_trapdoors:
                if not self.trapdoor_belief.is_confirmed_trapdoor(trap_loc):
                    self.trapdoor_belief.mark_confirmed_trapdoor(trap_loc)
        
        # Get trapdoor risk map
        trapdoor_risks = self._get_trapdoor_risk_map()
        
        # Initialize egg tour if not done yet (first turn)
        if not self.egg_tour and board.turn_count == 0:
            self.egg_tour = self.pathfinder.compute_egg_tour(board, self.my_parity, trapdoor_risks)
            # Try to optimize with 2-opt (time permitting)
            if time_left() > 5.0:
                self.egg_tour = self.pathfinder.optimize_tour_2opt(
                    self.egg_tour, board, trapdoor_risks, max_iterations=20
                )
        
        # Re-plan egg tour if enemy has blocked our path with turds
        if self.egg_tour and board.turn_count > 10:
            # Check if next target in tour is now blocked
            current_target_blocked = False
            if self.tour_index < len(self.egg_tour):
                target = self.egg_tour[self.tour_index]
                if target in board.turds_enemy or target in board.eggs_enemy:
                    current_target_blocked = True
            
            # Recompute tour every 10 turns or if target is blocked
            if current_target_blocked or board.turn_count % 10 == 0:
                self.egg_tour = self.pathfinder.compute_egg_tour(board, self.my_parity, trapdoor_risks)
                self.tour_index = 0
        
        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return (Direction.UP, MoveType.PLAIN)  # Shouldn't happen but just in case
        
        # AGGRESSIVE EARLY GAME: Rush eggs before enemy gets established
        # Greedy mode until turn 18 to collect eggs quickly
        if board.turn_count < 18:
            # Greedy: pick the egg move that gets us closest to an uncollected egg square
            egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
            if egg_moves:
                # Always lay egg if we can - but check trapdoor risk first
                safe_egg_moves = []
                for move in egg_moves:
                    next_loc = loc_after_direction(my_loc, move[0])
                    risk = trapdoor_risks.get(next_loc, 0.0)
                    if risk < 0.15:  # Safe enough
                        safe_egg_moves.append(move)
                
                if safe_egg_moves:
                    # Prefer corners (worth 3 eggs)
                    corner_moves = []
                    for move in safe_egg_moves:
                        if my_loc in [(0,0), (0,7), (7,0), (7,7)]:
                            corner_moves.append(move)
                    best_move = corner_moves[0] if corner_moves else safe_egg_moves[0]
                else:
                    # All egg moves are risky, fall back to minimax
                    should_turd = False
                    time_budget = self._calculate_time_budget(time_left(), turns_remaining)
                    best_move = self._iterative_deepening_search(
                        board, valid_moves, time_budget, start_time, should_turd
                    )
            else:
                # Move toward nearest egg square using A* path
                nearest_egg = self._find_nearest_egg_square(board, my_loc, self.my_parity)
                if nearest_egg:
                    # Use pathfinder for safer route
                    path = self.pathfinder.astar_path(board, my_loc, nearest_egg, trapdoor_risks)
                    if path and len(path) > 1:
                        # Get direction to next step in path
                        next_step = path[1]
                        direction = self._get_direction_to(my_loc, next_step)
                        if direction:
                            # Check if this direction is in valid moves
                            plain_move = (direction, MoveType.PLAIN)
                            if plain_move in valid_moves:
                                best_move = plain_move
                            else:
                                # Try egg move in that direction
                                egg_move = (direction, MoveType.EGG)
                                if egg_move in valid_moves:
                                    best_move = egg_move
                                else:
                                    best_move = self._greedy_move_toward(board, valid_moves, my_loc, nearest_egg, trapdoor_risks)
                        else:
                            best_move = self._greedy_move_toward(board, valid_moves, my_loc, nearest_egg, trapdoor_risks)
                    else:
                        best_move = self._greedy_move_toward(board, valid_moves, my_loc, nearest_egg, trapdoor_risks)
                else:
                    # No egg squares left, use normal search
                    should_turd = self.strategy.should_lay_turd_now(board, trapdoor_risks)
                    time_budget = self._calculate_time_budget(time_left(), turns_remaining)
                    best_move = self._iterative_deepening_search(
                        board, valid_moves, time_budget, start_time, should_turd
                    )
            
            self.history_table[best_move] += 1
            return best_move
        
        # Normal game: use minimax search
        # Quick strategic decision: should we lay turd?
        should_turd = self.strategy.should_lay_turd_now(board, trapdoor_risks)
        
        # Determine time budget for this move
        turns_remaining = board.turns_left_player
        time_remaining = time_left()
        time_budget = self._calculate_time_budget(time_remaining, turns_remaining)
        
        # Search for best move using iterative deepening minimax
        best_move = self._iterative_deepening_search(
            board, valid_moves, time_budget, start_time, should_turd
        )
        
        # Update history table with chosen move
        self.history_table[best_move] += 1
        
        elapsed = time.time() - start_time
        
        # Optional: log performance (can remove for submission)
        if False:  # Set to True for debugging
            print(f"[Magnus] Turn {board.turn_count}: {best_move}, "
                  f"{elapsed:.3f}s, {self.nodes_searched} nodes, "
                  f"{self.cache_hits} cache hits")
        
        return best_move
    
    def _calculate_time_budget(self, time_remaining: float, turns_remaining: int) -> float:
        """
        Calculate how much time to spend on this move.
        Conservative: use less time per move than average to avoid timeout.
        """
        if turns_remaining <= 0:
            return 0.1
        
        # Use 40% of average time per move (more conservative, faster moves)
        avg_time = time_remaining / turns_remaining
        budget = avg_time * 0.4
        
        # Minimum 0.05s, maximum 1.5s
        budget = max(0.05, min(1.5, budget))
        
        # In endgame with plenty of time, can search longer
        if turns_remaining <= 10 and time_remaining > 60:
            budget = min(2.0, budget * 1.5)
        
        return budget
    
    def _get_trapdoor_risk_map(self) -> Dict[Tuple[int, int], float]:
        """Get dictionary of trapdoor probabilities for all squares."""
        risks = {}
        for x in range(self.map_size):
            for y in range(self.map_size):
                loc = (x, y)
                risks[loc] = self.trapdoor_belief.get_prob_at(loc)
        return risks
    
    def _iterative_deepening_search(self, board: Board, valid_moves: List[Tuple[Direction, MoveType]],
                                   time_budget: float, start_time: float,
                                   prefer_turd: bool) -> Tuple[Direction, MoveType]:
        """
        Iterative deepening minimax with alpha-beta pruning.
        """
        # FILTER OUT DANGEROUS MOVES IMMEDIATELY (trapdoor risk > 0.15)
        trapdoor_risks = self._get_trapdoor_risk_map()
        safe_moves = []
        for move in valid_moves:
            next_board = board.forecast_move(move[0], move[1], check_ok=False)
            if next_board:
                next_loc = next_board.chicken_player.get_location()
                risk = trapdoor_risks.get(next_loc, 0.0)
                if risk < 0.15:  # Only consider moves to safe squares
                    safe_moves.append(move)
        
        # If all moves are risky, pick the least risky
        if not safe_moves:
            safe_moves = sorted(valid_moves, key=lambda m: trapdoor_risks.get(
                board.forecast_move(m[0], m[1], check_ok=False).chicken_player.get_location() if board.forecast_move(m[0], m[1], check_ok=False) else (0,0), 1.0
            ))[:max(1, len(valid_moves) // 2)]
        
        best_move = safe_moves[0] if safe_moves else valid_moves[0]  # Fallback
        best_score = -float('inf')
        
        # Order moves for better alpha-beta pruning
        ordered_moves = self._order_moves(board, safe_moves, depth=0, prefer_turd=prefer_turd)
        
        # Iterative deepening: depth 1, 2, 3, ...
        for depth in range(1, self.max_search_depth + 1):
            # Check time
            elapsed = time.time() - start_time
            if elapsed >= time_budget - self.time_safety_margin:
                break
            
            try:
                score, move = self._minimax(
                    board, depth, -float('inf'), float('inf'), True, 
                    start_time, time_budget, prefer_turd
                )
                
                if move is not None:
                    best_move = move
                    best_score = score
                
                # If we found a winning move, no need to search deeper
                if score > 50000:  # Very high score indicates likely win
                    break
                    
            except TimeoutError:
                break  # Time's up
        
        return best_move
    
    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, 
                maximizing: bool, start_time: float, time_budget: float,
                prefer_turd: bool) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        """
        Minimax search with alpha-beta pruning and transposition table.
        """
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed >= time_budget - self.time_safety_margin:
            raise TimeoutError()
        
        self.nodes_searched += 1
        
        # Check transposition table
        board_hash = self._hash_board(board)
        if board_hash in self.transposition_table:
            cached_score, cached_move, cached_depth = self.transposition_table[board_hash]
            if cached_depth >= depth:
                self.cache_hits += 1
                return cached_score, cached_move
        
        # Terminal node or depth limit
        if depth == 0 or board.is_game_over():
            score = self._evaluate(board)
            return score, None
        
        # Get valid moves
        moves = board.get_valid_moves(enemy=not maximizing)
        if not moves:
            # No valid moves = we lose (or opponent loses)
            score = -100000 if maximizing else 100000
            return score, None
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(board, moves, depth, prefer_turd and maximizing)
        
        best_move = None
        
        if maximizing:
            max_eval = -float('inf')
            
            for move in ordered_moves:
                # Forecast move
                next_board = board.forecast_move(move[0], move[1], check_ok=False)
                if next_board is None:
                    continue
                
                # Recursive search
                eval_score, _ = self._minimax(
                    next_board, depth - 1, alpha, beta, False, 
                    start_time, time_budget, prefer_turd
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    # Killer move heuristic
                    if best_move and best_move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, best_move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth] = self.killer_moves[depth][:2]
                    break
            
            # Cache result
            self.transposition_table[board_hash] = (max_eval, best_move, depth)
            return max_eval, best_move
        
        else:  # Minimizing (opponent's turn)
            min_eval = float('inf')
            
            for move in ordered_moves:
                next_board = board.forecast_move(move[0], move[1], check_ok=False)
                if next_board is None:
                    continue
                
                eval_score, _ = self._minimax(
                    next_board, depth - 1, alpha, beta, True,
                    start_time, time_budget, prefer_turd
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    if best_move and best_move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, best_move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth] = self.killer_moves[depth][:2]
                    break
            
            self.transposition_table[board_hash] = (min_eval, best_move, depth)
            return min_eval, best_move
    
    def _evaluate(self, board: Board) -> float:
        """
        Evaluation function for board position.
        Higher score = better for us.
        """
        my_chicken = board.chicken_player
        enemy_chicken = board.chicken_enemy
        my_loc = my_chicken.get_location()
        enemy_loc = enemy_chicken.get_location()
        
        score = 0.0
        
        # 0. ANTI-BARRIER STRATEGY (early game priority)
        # Detect if enemy is building a diagonal barrier
        if board.turn_count < 25:
            barrier_detected = self.strategy.detect_barrier_strategy(board)
            if barrier_detected:
                # CRITICAL: Stay on the enemy side of the barrier, not our side
                my_spawn = my_chicken.get_spawn()
                enemy_spawn = enemy_chicken.get_spawn()
                
                if my_spawn and enemy_spawn and self.strategy.barrier_diagonal:
                    # Determine which side of spawn line we're on
                    dx, dy = self.strategy.barrier_diagonal
                    
                    # Calculate if we're advancing toward enemy or retreating to our spawn
                    # We want to advance PAST the forming barrier, not retreat behind it
                    if abs(enemy_spawn[0] - my_spawn[0]) > abs(enemy_spawn[1] - my_spawn[1]):
                        # Horizontal separation - check x advancement
                        direction = 1 if enemy_spawn[0] > my_spawn[0] else -1
                        advancement = (my_loc[0] - my_spawn[0]) * direction
                    else:
                        # Vertical separation - check y advancement  
                        direction = 1 if enemy_spawn[1] > my_spawn[1] else -1
                        advancement = (my_loc[1] - my_spawn[1]) * direction
                    
                    # HUGE bonus for being on enemy side (past center)
                    if advancement > 2:
                        score += advancement * 500  # Massive reward
                    else:
                        # Penalty for being trapped on our side
                        score -= (3 - advancement) * 800
        
        # 1. EGG DIFFERENTIAL (most important)
        my_eggs = my_chicken.get_eggs_laid()
        enemy_eggs = enemy_chicken.get_eggs_laid()
        egg_diff = my_eggs - enemy_eggs
        
        # In early game (< turn 20), eggs are CRITICAL - Cordelia ignores them
        if board.turn_count < 20:
            score += egg_diff * 2000  # Double importance early
        else:
            score += egg_diff * 1000
        
        # 2. MOBILITY (valid moves available)
        my_moves = len(board.get_valid_moves(enemy=False))
        enemy_moves = len(board.get_valid_moves(enemy=True))
        score += (my_moves - enemy_moves) * 300
        
        # If opponent has no moves, huge bonus (we win)
        if enemy_moves == 0:
            score += 100000
        if my_moves == 0:
            score -= 100000
        
        # 3. TRAPDOOR RISK at our position (CRITICAL - heavily penalize risky squares)
        trap_risk = self.trapdoor_belief.get_prob_at(my_loc)
        score -= trap_risk * 10000  # Very high penalty - stepping on trap gives enemy 4 eggs
        
        # 4. CORNER PROXIMITY/CONTROL (corners worth 3 eggs)
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for corner in corners:
            corner_parity = (corner[0] + corner[1]) % 2
            
            # Our corner
            if corner_parity == self.my_parity and corner not in board.eggs_player:
                dist_me = abs(my_loc[0] - corner[0]) + abs(my_loc[1] - corner[1])
                dist_enemy = abs(enemy_loc[0] - corner[0]) + abs(enemy_loc[1] - corner[1])
                
                # Bonus for being closer to unvisited corner
                if dist_me < dist_enemy:
                    score += (6 - dist_me) * 100
                
                # Huge bonus if we're ON the corner
                if my_loc == corner:
                    score += 1000
        
        # 5. DISTANCE TO NEAREST EGG SQUARE
        nearest_egg_sq_dist = self._nearest_egg_square_distance(board, my_loc, self.my_parity)
        if nearest_egg_sq_dist is not None:
            score -= nearest_egg_sq_dist * 20
        
        # 6. TURD ADVANTAGE
        my_turds = my_chicken.get_turds_left()
        enemy_turds = enemy_chicken.get_turds_left()
        score += (my_turds - enemy_turds) * 50
        
        # 7. CENTER CONTROL (center squares are more flexible)
        center_x, center_y = 3.5, 3.5
        my_center_dist = abs(my_loc[0] - center_x) + abs(my_loc[1] - center_y)
        enemy_center_dist = abs(enemy_loc[0] - center_x) + abs(enemy_loc[1] - center_y)
        score -= my_center_dist * 15
        score += enemy_center_dist * 10
        
        # 8. DISTANCE TO ENEMY (slight aggression)
        enemy_dist = abs(my_loc[0] - enemy_loc[0]) + abs(my_loc[1] - enemy_loc[1])
        score -= enemy_dist * 5
        
        # 9. BARRIER COUNTER-STRATEGY
        # If enemy has built a turd wall, reward breaking through to their side
        if len(board.turds_enemy) >= 3:
            # Check if we're on "enemy side" (past their turd line)
            enemy_spawn = enemy_chicken.get_spawn()
            my_spawn = my_chicken.get_spawn()
            if enemy_spawn and my_spawn:
                # Determine axis (horizontal or vertical)
                if abs(enemy_spawn[0] - my_spawn[0]) > abs(enemy_spawn[1] - my_spawn[1]):
                    # Horizontal separation
                    direction = 1 if enemy_spawn[0] > my_spawn[0] else -1
                    penetration = (my_loc[0] - my_spawn[0]) * direction
                    score += penetration * 200  # Reward for penetrating enemy territory
                else:
                    # Vertical separation
                    direction = 1 if enemy_spawn[1] > my_spawn[1] else -1
                    penetration = (my_loc[1] - my_spawn[1]) * direction
                    score += penetration * 200
        
        return score
    
    def _count_reachable_squares(self, board: Board, start_loc: Tuple[int, int], 
                                 depth: int = 5) -> int:
        """
        Count how many unique squares are reachable from start_loc within depth moves.
        This measures freedom/mobility.
        """
        visited = {start_loc}
        frontier = [start_loc]
        
        for _ in range(depth):
            next_frontier = []
            for loc in frontier:
                for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                    next_loc = loc_after_direction(loc, direction)
                    
                    # Check bounds
                    if not (0 <= next_loc[0] < self.map_size and 
                           0 <= next_loc[1] < self.map_size):
                        continue
                    
                    # Already visited
                    if next_loc in visited:
                        continue
                    
                    # Check if move is valid (use board's validation)
                    # Simple check: not enemy position, not enemy egg/turd, not adjacent to enemy turd
                    if next_loc == board.chicken_enemy.get_location():
                        continue
                    if next_loc in board.eggs_enemy or next_loc in board.turds_enemy:
                        continue
                    
                    # Check adjacency to enemy turds
                    is_adjacent_to_enemy_turd = False
                    for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                        adj_loc = loc_after_direction(next_loc, d)
                        if adj_loc in board.turds_enemy:
                            is_adjacent_to_enemy_turd = True
                            break
                    
                    if is_adjacent_to_enemy_turd:
                        continue
                    
                    # Valid square
                    visited.add(next_loc)
                    next_frontier.append(next_loc)
            
            frontier = next_frontier
            if not frontier:
                break
        
        return len(visited)
    
    def _find_nearest_egg_square(self, board: Board, loc: Tuple[int, int], 
                                 parity: int) -> Optional[Tuple[int, int]]:
        """Find nearest uncollected egg square with correct parity."""
        min_dist = float('inf')
        nearest = None
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                if (x + y) % 2 != parity:
                    continue
                
                sq = (x, y)
                if sq in board.eggs_player or sq in board.eggs_enemy:
                    continue
                
                dist = abs(loc[0] - x) + abs(loc[1] - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = sq
        
        return nearest
    
    def _greedy_move_toward(self, board: Board, valid_moves: List[Tuple[Direction, MoveType]],
                           current_loc: Tuple[int, int], target: Tuple[int, int],
                           trapdoor_risks: Dict[Tuple[int, int], float]) -> Tuple[Direction, MoveType]:
        """Greedy move selection: move toward target while avoiding trapdoors."""
        best_move = valid_moves[0]
        best_dist = float('inf')
        
        for move in valid_moves:
            direction, move_type = move
            
            # Skip turd moves in greedy mode
            if move_type == MoveType.TURD:
                continue
            
            # Calculate where we'd be after this move
            next_loc = loc_after_direction(current_loc, direction)
            
            # Skip if high trapdoor risk
            if next_loc in trapdoor_risks and trapdoor_risks[next_loc] > 0.15:
                continue
            
            # Calculate distance to target
            dist = abs(next_loc[0] - target[0]) + abs(next_loc[1] - target[1])
            
            # Prefer moves that get us closer
            if dist < best_dist:
                best_dist = dist
                best_move = move
        
        return best_move
    
    def _get_direction_to(self, from_loc: Tuple[int, int], to_loc: Tuple[int, int]) -> Optional[Direction]:
        """Get the direction from one square to an adjacent square."""
        dx = to_loc[0] - from_loc[0]
        dy = to_loc[1] - from_loc[1]
        
        if dx == 0 and dy == -1:
            return Direction.UP
        elif dx == 0 and dy == 1:
            return Direction.DOWN
        elif dx == -1 and dy == 0:
            return Direction.LEFT
        elif dx == 1 and dy == 0:
            return Direction.RIGHT
        else:
            return None  # Not adjacent
    
    def _nearest_egg_square_distance(self, board: Board, loc: Tuple[int, int], 
                                    parity: int) -> Optional[int]:
        """Find manhattan distance to nearest unvisited egg square."""
        min_dist = None
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                if (x + y) % 2 != parity:
                    continue
                
                sq = (x, y)
                if sq in board.eggs_player or sq in board.eggs_enemy:
                    continue
                
                dist = abs(loc[0] - x) + abs(loc[1] - y)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
        
        return min_dist
    
    def _order_moves(self, board: Board, moves: List[Tuple[Direction, MoveType]], 
                    depth: int, prefer_turd: bool = False) -> List[Tuple[Direction, MoveType]]:
        """
        Order moves for better alpha-beta pruning.
        Better moves first = more cutoffs.
        """
        def move_priority(move: Tuple[Direction, MoveType]) -> Tuple[int, int, int]:
            direction, move_type = move
            
            # Priority 1: Killer moves (moves that caused cutoffs at this depth)
            killer_bonus = 0
            if move in self.killer_moves[depth]:
                killer_bonus = -1000 + self.killer_moves[depth].index(move)
            
            # Priority 2: Move type
            # In early game (< turn 20), heavily prioritize egg moves
            if board.turn_count < 20 and move_type == MoveType.EGG:
                type_priority = -1  # Highest priority
            elif prefer_turd and move_type == MoveType.TURD:
                type_priority = 0
            elif move_type == MoveType.EGG:
                type_priority = 1
            elif move_type == MoveType.TURD:
                type_priority = 2
            else:
                type_priority = 3
            
            # Priority 3: History heuristic (moves that were good in past)
            history_score = -self.history_table.get(move, 0)
            
            return (killer_bonus, type_priority, history_score)
        
        return sorted(moves, key=move_priority)
    
    def _hash_board(self, board: Board) -> int:
        """
        Create a hash of the board state for transposition table.
        """
        # Simple hash based on key positions and egg counts
        h = 0
        h ^= hash(board.chicken_player.get_location())
        h ^= hash(board.chicken_enemy.get_location()) << 3
        h ^= board.chicken_player.get_eggs_laid() << 6
        h ^= board.chicken_enemy.get_eggs_laid() << 10
        h ^= hash(frozenset(board.eggs_player)) << 14
        h ^= hash(frozenset(board.eggs_enemy)) << 18
        h ^= hash(frozenset(board.turds_player)) << 22
        h ^= hash(frozenset(board.turds_enemy)) << 26
        return h
