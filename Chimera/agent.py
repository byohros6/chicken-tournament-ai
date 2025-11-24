"""
Project Chimera: The "God Agent"
Combines Magnus's Engine & Pathfinding with Cordelia's Strategy & Phases.
Features:
- Adaptive Pruning (Hard prune safe moves, Soft prune forced risks)
- Phase-Weighted Heuristics (Opening Wall -> Midgame Tour -> Endgame Survival)
- A* Egg Tours
"""
import random
import time
import math
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, deque

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction

# Import Magnus's superior modules
from .trapdoor_belief import TrapdoorBelief
from .pathfinding import PathFinder
from .strategy import StrategicPlanner

class PlayerAgent:
    def __init__(self, board: Board, time_left):
        # --- COMPONENT 1: Magnus's Engine ---
        self.trap_belief = TrapdoorBelief(board.game_map.MAP_SIZE)
        self.pathfinder = PathFinder(board)
        self.strategy = StrategicPlanner(board)
        
        # --- COMPONENT 2: Search Optimization ---
        self.transposition_table = {}
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.recent_locs = deque(maxlen=8)
        self.escape_target: Optional[Tuple[int, int]] = None
        self.current_target: Optional[Tuple[int, int]] = None
        self.last_tour_refresh = 0
        self.force_turd = False
        
        # --- COMPONENT 3: Long-Term Memory (The Tour) ---
        self.egg_tour: List[Tuple[int, int]] = []
        self.tour_index = 0
        self.my_parity = board.chicken_player.even_chicken
        self.move_history = deque(maxlen=6)  # Track last 6 moves for oscillation detection
        
        # Constants
        self.MAX_DEPTH = 12
        self.TIME_BUFFER = 0.05

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left) -> Tuple[Direction, MoveType]:
        start_time = time.time()
        my_loc = board.chicken_player.get_location()

        # 1. Update beliefs and trap cache
        self.trap_belief.update(my_loc, sensor_data)
        trap_risks = self._get_trap_risks()
        self.recent_locs.append(my_loc)
        self.strategy.detect_barrier_strategy(board)

        # 2. Refresh egg tour / escape plans
        if self._needs_tour_refresh(board):
            self.egg_tour = self.pathfinder.compute_egg_tour(board, self.my_parity, trap_risks)
            self.tour_index = 0
            self.last_tour_refresh = board.turn_count
        if self.tour_index < len(self.egg_tour) and my_loc == self.egg_tour[self.tour_index]:
            self.tour_index += 1

        stuck = self._is_stuck()
        if stuck and not self.escape_target:
            self.escape_target = self._select_escape_target(board, trap_risks)
        if not stuck and self.escape_target and my_loc != self.escape_target:
            # Release escape goal once we have fresh mobility
            self.escape_target = None
        if self.escape_target and my_loc == self.escape_target:
            self.escape_target = None

        phase = self.strategy.get_game_phase(board)
        target = self._select_primary_target(board, trap_risks, phase)
        barrier_target = None
        if target is None:
            barrier_target = self.strategy.get_barrier_objective(board, trap_risks)
        self.current_target = barrier_target or target

        # 3. Determine time budget & turd policy
        turns_remaining = board.turns_left_player
        budget = self._calculate_time_budget(time_left(), turns_remaining)
        self.force_turd = self.strategy.should_lay_turd_now(board, trap_risks)

        # 4. Root move filtering with adaptive pruning
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return (Direction.UP, MoveType.PLAIN)

        # AGGRESSIVE STRATEGY: Magnus-style egg rush for ENTIRE GAME
        # Minimax is too slow and gets stuck - greedy egg collection wins!
        greedy_move = self._greedy_egg_rush(board, valid_moves, trap_risks)
        if greedy_move:
            self.history_table[greedy_move] += 1
            # Track move history for oscillation detection
            if greedy_move[1] != MoveType.TURD:
                next_loc = loc_after_direction(my_loc, greedy_move[0])
                self.move_history.append(next_loc)
            return greedy_move

        search_moves = self._adaptive_prune(board, valid_moves, trap_risks)
        if not search_moves:
            search_moves = valid_moves

        # ANTI-OSCILLATION for minimax: Avoid repeating same locations
        if len(self.move_history) >= 6:
            recent_locs = list(self.move_history)[-6:]
            unique_recent = len(set(recent_locs))
            if unique_recent <= 2:  # Stuck in tight loop
                # Filter search space to exclude recent locations
                escape_moves = []
                for move in search_moves:
                    if move[1] == MoveType.TURD:
                        next_loc = my_loc
                    else:
                        next_loc = loc_after_direction(my_loc, move[0])
                    if next_loc not in recent_locs:
                        escape_moves.append(move)
                if escape_moves:
                    # FORCE escape immediately - don't trust minimax to pick correctly!
                    # Prefer egg moves if available
                    egg_escapes = [m for m in escape_moves if m[1] == MoveType.EGG]
                    if egg_escapes:
                        chosen_escape = egg_escapes[0]
                        # Track the escape location
                        if chosen_escape[1] != MoveType.TURD:
                            next_loc = loc_after_direction(my_loc, chosen_escape[0])
                            self.move_history.append(next_loc)
                        return chosen_escape
                    else:
                        chosen_escape = escape_moves[0]
                        # Track the escape location
                        if chosen_escape[1] != MoveType.TURD:
                            next_loc = loc_after_direction(my_loc, chosen_escape[0])
                            self.move_history.append(next_loc)
                        return chosen_escape
                else:
                    # If ALL moves lead to recent locations, LAY A TURD to break the cycle!
                    turd_moves = [m for m in search_moves if m[1] == MoveType.TURD]
                    if turd_moves:
                        # Don't append to move_history for turds (we stay at same location)
                        return turd_moves[0]  # Immediately return - FORCE break oscillation

        search_moves = self._order_moves(board, search_moves, depth=0)
        best_move = search_moves[0]
        best_score = -float('inf')

        # 5. Iterative deepening search
        try:
            for depth in range(1, self.MAX_DEPTH + 1):
                if time.time() - start_time > budget:
                    break

                score, move = self._minimax(
                    board, depth, -float('inf'), float('inf'), True,
                    start_time, budget, phase
                )

                if move:
                    best_move = move
                    best_score = score
                    if score > 90000:
                        break

        except TimeoutError:
            pass

        self.history_table[best_move] += 1
        
        # Track move history for oscillation detection
        if best_move[1] != MoveType.TURD:
            next_loc = loc_after_direction(my_loc, best_move[0])
            self.move_history.append(next_loc)
        
        return best_move

    # ------------------------------------------------------------------
    # The Search Engine (Magnus's Core)
    # ------------------------------------------------------------------
    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, 
                 is_max: bool, start_time: float, budget: float, phase: str):
        
        # Timeout Check
        if (self.transposition_table and # Don't check every node
            time.time() - start_time > budget - self.TIME_BUFFER):
            raise TimeoutError()

        # Transposition Table Lookup
        board_hash = self._hash_board(board)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depth:
                return entry['score'], entry['move']

        if depth == 0 or board.is_game_over():
            return self._evaluate_hybrid(board, phase), None

        # Move Generation & Ordering
        moves = board.get_valid_moves(enemy=not is_max)
        if not moves:
            return (-100000 if is_max else 100000), None
            
        # Optimization: Only order moves at higher depths
        if depth > 1:
            moves = self._order_moves(board, moves, depth, is_max)

        best_move = moves[0]
        
        if is_max:
            max_eval = -float('inf')
            for move in moves:
                next_b = board.forecast_move(move[0], move[1], check_ok=False)
                if not next_b: continue
                
                score, _ = self._minimax(next_b, depth-1, alpha, beta, False, start_time, budget, phase)
                
                if score > max_eval:
                    max_eval = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    self._store_killer(depth, move)
                    break
            
            self._store_tt(board_hash, max_eval, best_move, depth)
            return max_eval, best_move
            
        else: # Minimizing (Enemy)
            min_eval = float('inf')
            for move in moves:
                next_b = board.forecast_move(move[0], move[1], check_ok=False)
                if not next_b: continue
                
                score, _ = self._minimax(next_b, depth-1, alpha, beta, True, start_time, budget, phase)
                
                if score < min_eval:
                    min_eval = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    self._store_killer(depth, move)
                    break
            
            self._store_tt(board_hash, min_eval, best_move, depth)
            return min_eval, best_move

    # ------------------------------------------------------------------
    # The Brain: Phase-Weighted Hybrid Heuristic
    # ------------------------------------------------------------------
    def _evaluate_hybrid(self, board: Board, phase: str) -> float:
        """
        The 'God Agent' Heuristic.
        Combines Cordelia's phase goals with Magnus's pathfinding and trap safety.
        """
        me = board.chicken_player
        en = board.chicken_enemy
        my_loc = me.get_location()
        en_loc = en.get_location()
        
        egg_diff = me.get_eggs_laid() - en.get_eggs_laid()
        score = egg_diff * 1500  # Increased from 1000

        # ANTI-OSCILLATION: BRUTAL penalty for revisiting recent squares
        if my_loc in list(self.move_history)[-6:]:
            repetitions = list(self.move_history)[-6:].count(my_loc)
            score -= repetitions * 10000  # MASSIVE penalty - worse than losing 6 eggs!

        risk = self.trap_belief.get_prob_at(my_loc)
        trap_weight = 15000 if phase != 'endgame' else 12000  # Increased penalties
        score -= risk * trap_weight

        my_moves = len(board.get_valid_moves(enemy=False))
        enemy_moves = len(board.get_valid_moves(enemy=True))
        score += (my_moves - enemy_moves) * 120  # Increased from 80
        if enemy_moves <= 2:
            score += 1500  # Increased from 1000
        if my_moves <= 1:
            score -= 1500  # Increased from 1200

        if self.current_target:
            target = self.current_target
            dist = abs(my_loc[0] - target[0]) + abs(my_loc[1] - target[1])
            score -= dist * 150  # Increased from 120
            if dist == 0:
                if hasattr(board, 'can_lay_egg_at_loc') and board.can_lay_egg_at_loc(my_loc):
                    score += 600  # Increased from 400
                else:
                    score += 300  # Increased from 200

        if self.escape_target:
            dist_escape = abs(my_loc[0] - self.escape_target[0]) + abs(my_loc[1] - self.escape_target[1])
            score -= dist_escape * 50  # Increased from 30

        if self.strategy.detected_barrier:
            if self._is_behind_barrier(board, my_loc):
                score -= 800  # Increased from 500
            else:
                score += 300  # Increased from 200

        if phase == 'early':
            score += self._corner_value(my_loc) * 60  # Increased from 40
        elif phase == 'mid':
            score += len(board.turds_player) * 40  # Increased from 30
            score -= len(board.turds_enemy) * 35  # Increased from 25
        else:
            dist_en = abs(my_loc[0] - en_loc[0]) + abs(my_loc[1] - en_loc[1])
            if egg_diff > 0:
                score += dist_en * 80  # Increased from 60
            else:
                score -= dist_en * 90  # Increased from 70
            score -= len(board.turds_enemy) * 30  # Increased from 20

        return score

    # ------------------------------------------------------------------
    # Helper Functions (Pruning, Sorting, Time)
    # ------------------------------------------------------------------
    def _adaptive_prune(self, board, moves, trap_risks):
        """
        Adaptive pruning with dynamic trap thresholds and escape overrides.
        """
        safe_moves: List[Tuple[Direction, MoveType]] = []
        neutral_moves: List[Tuple[Direction, MoveType]] = []
        risky_moves: List[Tuple[Direction, MoveType]] = []

        threshold = self._get_dynamic_risk_threshold(board)
        desperation = min(0.45, threshold + 0.25)

        for move in moves:
            nxt = board.forecast_move(move[0], move[1], check_ok=False)
            if not nxt:
                continue
            loc = nxt.chicken_player.get_location()
            risk = trap_risks.get(loc, 0.0)

            if self.escape_target and loc == self.escape_target:
                safe_moves.append(move)
                continue

            if risk < threshold:
                safe_moves.append(move)
            elif risk < desperation:
                neutral_moves.append(move)
            else:
                risky_moves.append(move)

        if len(safe_moves) >= 2:
            return safe_moves
        if safe_moves:
            return safe_moves + neutral_moves
        if neutral_moves:
            return neutral_moves
        return risky_moves

    def _get_trap_risks(self):
        risks = {}
        for x in range(8):
            for y in range(8):
                risks[(x,y)] = self.trap_belief.get_prob_at((x,y))
        return risks

    def _calculate_time_budget(self, time_left, turns_left):
        if turns_left <= 0: return 0.1
        # Reserve buffer, spend more in mid-game
        base = time_left / turns_left
        return max(0.1, min(1.5, base * 0.6))

    def _order_moves(self, board, moves, depth, is_max=True):
        actor = board.chicken_player if is_max else board.chicken_enemy
        curr_loc = actor.get_location()
        enemy_loc = board.chicken_enemy.get_location() if is_max else board.chicken_player.get_location()

        def score_move(move):
            score = 0
            if move in self.killer_moves[depth]:
                score += 10000
            score += self.history_table[move]

            move_type = move[1]
            if move_type == MoveType.EGG:
                score += 150
            elif move_type == MoveType.TURD:
                score += 200 if (is_max and self.force_turd) else -80

            # Calculate next location
            if move_type == MoveType.TURD:
                next_loc = curr_loc
            else:
                next_loc = loc_after_direction(curr_loc, move[0])

            # ANTI-OSCILLATION: Detect 2-4 move cycles
            if is_max and len(self.move_history) >= 2:
                # Check if this move creates a 2-move loop (A->B->A)
                if len(self.move_history) >= 1 and self.move_history[-1] == next_loc:
                    score -= 8000
                # Check if this move creates a 3-move loop
                elif len(self.move_history) >= 2 and self.move_history[-2] == next_loc:
                    score -= 6000
                # Check if this move creates a 4-move loop
                elif len(self.move_history) >= 3 and self.move_history[-3] == next_loc:
                    score -= 4000

            # CORNER AWARENESS: Strong preference for corners when adjacent
            if is_max:
                corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
                if move_type == MoveType.PLAIN:
                    # HUGE bonus for moving TO a corner
                    if next_loc in corners:
                        score += 1500  # Increased from 800
                else:
                    # Strongly penalize laying egg when adjacent to corner
                    adjacent_to_corner = any(
                        abs(curr_loc[0] - c[0]) + abs(curr_loc[1] - c[1]) == 1
                        for c in corners
                    )
                    if adjacent_to_corner and move_type == MoveType.EGG:
                        score -= 1000  # Increased from 600

            # ENEMY AVOIDANCE: Avoid enemy's turd-infested areas
            if is_max and move_type == MoveType.PLAIN:
                enemy_turds = board.turds_enemy
                # Count enemy turds near next location
                nearby_enemy_turds = sum(
                    1 for turd in enemy_turds
                    if abs(next_loc[0] - turd[0]) + abs(next_loc[1] - turd[1]) <= 2
                )
                if nearby_enemy_turds > 0:
                    score -= nearby_enemy_turds * 300  # Penalty for being near enemy mess
                
                # Avoid moving too close to enemy (unless attacking)
                dist_to_enemy = abs(next_loc[0] - enemy_loc[0]) + abs(next_loc[1] - enemy_loc[1])
                if dist_to_enemy <= 2 and len(enemy_turds) >= 3:
                    score -= 400  # Stay away from enemy when they have turds

            if is_max and self.current_target:
                target = self.current_target
                curr_dist = abs(curr_loc[0] - target[0]) + abs(curr_loc[1] - target[1])
                if move_type == MoveType.TURD:
                    next_dist = curr_dist
                else:
                    next_dist = abs(next_loc[0] - target[0]) + abs(next_loc[1] - target[1])
                score += (curr_dist - next_dist) * 40

            return score

        return sorted(moves, key=score_move, reverse=True)

    def _store_tt(self, h, score, move, depth):
        self.transposition_table[h] = {'score': score, 'move': move, 'depth': depth}
        
    def _store_killer(self, depth, move):
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].append(move)
            
    def _hash_board(self, board):
        # Simple Zobrist-like hash for python
        return hash((
            board.chicken_player.get_location(),
            board.chicken_enemy.get_location(),
            frozenset(board.eggs_player),
            frozenset(board.turds_player)
        ))

    # ------------------------------------------------------------------
    # Targeting & State Helpers
    # ------------------------------------------------------------------
    def _needs_tour_refresh(self, board: Board) -> bool:
        if not self.egg_tour:
            return True
        if board.turn_count - self.last_tour_refresh >= 8:
            return True
        if self.tour_index >= len(self.egg_tour):
            return True
        next_target = self._next_tour_target(board)
        if next_target is None:
            return True
        if next_target in board.eggs_enemy or next_target in board.turds_enemy:
            return True
        return False

    def _next_tour_target(self, board: Board) -> Optional[Tuple[int, int]]:
        while self.tour_index < len(self.egg_tour):
            target = self.egg_tour[self.tour_index]
            if target in board.eggs_player:
                self.tour_index += 1
                continue
            if target in board.turds_enemy or target in board.eggs_enemy:
                self.tour_index += 1
                continue
            return target
        return None

    def _select_primary_target(self, board: Board, trap_risks: Dict[Tuple[int, int], float], phase: str) -> Optional[Tuple[int, int]]:
        if self.escape_target:
            return self.escape_target

        tour_target = self._next_tour_target(board)
        if tour_target:
            return tour_target

        bfs_target = self._bfs_target(board, trap_risks)
        if bfs_target:
            return bfs_target

        return self.strategy.get_strategic_target(board, self.egg_tour, trap_risks)

    def _bfs_target(self, board: Board, trap_risks: Dict[Tuple[int, int], float], max_depth: int = 10) -> Optional[Tuple[int, int]]:
        start = board.chicken_player.get_location()
        queue = deque([(start, 0)])
        visited = {start}
        size = board.game_map.MAP_SIZE
        while queue:
            loc, dist = queue.popleft()
            if loc != start and self._is_safe_square(board, loc, trap_risks):
                return loc
            if dist >= max_depth:
                continue
            for direction in Direction:
                next_loc = loc_after_direction(loc, direction)
                if not self._is_in_bounds(next_loc, size):
                    continue
                if next_loc in visited:
                    continue
                if next_loc in board.turds_player or next_loc in board.turds_enemy:
                    continue
                visited.add(next_loc)
                queue.append((next_loc, dist + 1))
        return None

    def _select_escape_target(self, board: Board, trap_risks: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        best = None
        best_dist = -1
        my_loc = board.chicken_player.get_location()
        size = board.game_map.MAP_SIZE
        for x in range(size):
            for y in range(size):
                loc = (x, y)
                if not self._is_safe_square(board, loc, trap_risks):
                    continue
                dist = abs(my_loc[0] - x) + abs(my_loc[1] - y)
                if dist > best_dist:
                    best_dist = dist
                    best = loc
        return best

    def _is_safe_square(self, board: Board, loc: Tuple[int, int], trap_risks: Dict[Tuple[int, int], float]) -> bool:
        if (loc[0] + loc[1]) % 2 != self.my_parity:
            return False
        if loc in board.eggs_player or loc in board.turds_player or loc in board.turds_enemy:
            return False
        if trap_risks.get(loc, 0.0) >= 0.25:
            return False
        return True

    def _is_in_bounds(self, loc: Tuple[int, int], size: int) -> bool:
        x, y = loc
        return 0 <= x < size and 0 <= y < size

    def _count_reachable(self, board: Board, start: Tuple[int, int], depth: int = 6, enemy: bool = False) -> int:
        """Count how many squares are reachable within depth moves (BFS mobility metric)."""
        size = board.game_map.MAP_SIZE
        queue = deque([(start, 0)])
        visited = {start}
        count = 0
        
        while queue:
            loc, dist = queue.popleft()
            count += 1
            if dist >= depth:
                continue
            
            for direction in Direction:
                next_loc = loc_after_direction(loc, direction)
                if not self._is_in_bounds(next_loc, size):
                    continue
                if next_loc in visited:
                    continue
                if next_loc in board.turds_player or next_loc in board.turds_enemy:
                    continue
                if not enemy and next_loc == board.chicken_enemy.get_location():
                    continue
                if enemy and next_loc == board.chicken_player.get_location():
                    continue
                visited.add(next_loc)
                queue.append((next_loc, dist + 1))
        
        return count

    def _is_stuck(self) -> bool:
        if len(self.recent_locs) < self.recent_locs.maxlen:
            return False
        return len(set(self.recent_locs)) <= 3

    def _get_dynamic_risk_threshold(self, board: Board) -> float:
        base = 0.18  # Increased from 0.15 to be slightly more cautious
        egg_deficit = board.chicken_enemy.get_eggs_laid() - board.chicken_player.get_eggs_laid()
        if egg_deficit >= 2:
            base += min(0.08, 0.03 * egg_deficit)  # More aggressive when behind
        if board.turns_left_player <= 6:
            base += 0.06  # Increased from 0.04
        if self.strategy.get_game_phase(board) == 'endgame':
            base += 0.03  # Increased from 0.02
        return min(0.35, base)

    def _corner_value(self, loc: Tuple[int, int]) -> float:
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        if loc in corners and (loc[0] + loc[1]) % 2 == self.my_parity:
            return 12.0  # Increased from 8.0
        best = min(abs(loc[0] - c[0]) + abs(loc[1] - c[1]) for c in corners)
        return max(0.0, 8 - best)  # Increased from 6 - best

    def _is_behind_barrier(self, board: Board, my_loc: Tuple[int, int]) -> bool:
        diag_sum = board.game_map.MAP_SIZE - 1
        my_sum = my_loc[0] + my_loc[1]
        enemy_sum = board.chicken_enemy.get_location()[0] + board.chicken_enemy.get_location()[1]
        if enemy_sum >= diag_sum and my_sum <= diag_sum - 2:
            return True
        if enemy_sum <= diag_sum and my_sum >= diag_sum + 2:
            return True
        return False

    def _greedy_egg_rush(self, board: Board, valid_moves: List[Tuple[Direction, MoveType]], 
                         trap_risks: Dict[Tuple[int, int], float]) -> Optional[Tuple[Direction, MoveType]]:
        """
        Magnus-style aggressive early game: prioritize laying eggs and rushing corners.
        WITH ANTI-OSCILLATION: Avoid revisiting recent locations.
        """
        my_loc = board.chicken_player.get_location()
        
        # Get egg counts for dynamic risk assessment
        my_eggs = board.chicken_player.get_eggs_laid()
        opp_eggs = board.chicken_enemy.get_eggs_laid()
        
        # ANTI-OSCILLATION: Check if we're stuck in a loop
        if len(self.move_history) >= 6:
            recent_locs = list(self.move_history)[-6:]
            unique_recent = len(set(recent_locs))
            if unique_recent <= 2:  # Only 1-2 locations in last 6 moves = STUCK
                # Filter out moves that return to recent locations
                escape_moves = []
                for move in valid_moves:
                    if move[1] == MoveType.TURD:
                        next_loc = my_loc
                    else:
                        next_loc = loc_after_direction(my_loc, move[0])
                    if next_loc not in recent_locs:
                        escape_moves.append(move)
                if escape_moves:
                    # Prefer egg moves, then plain moves
                    egg_escapes = [m for m in escape_moves if m[1] == MoveType.EGG]
                    if egg_escapes:
                        return egg_escapes[0]
                    return escape_moves[0]
                else:
                    # TRAPPED! All moves lead back to recent locations
                    # Lay a turd to block the oscillation path
                    turd_moves = [m for m in valid_moves if m[1] == MoveType.TURD]
                    if turd_moves:
                        return turd_moves[0]
                    # No turds left - just pick first valid move and hope
                    return valid_moves[0] if valid_moves else None
        
        # Always lay egg if safe
        egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
        safe_egg_moves = []
        
        # DYNAMIC RISK FOR EGG LAYING: Be aggressive when behind
        egg_diff = my_eggs - opp_eggs
        if egg_diff < -3:  # Significantly behind - TAKE RISKS
            max_egg_risk = 0.40
        elif egg_diff < 0:  # Behind - accept moderate risk
            max_egg_risk = 0.32
        else:  # Ahead or tied - be cautious
            max_egg_risk = 0.25
        
        for move in egg_moves:
            next_loc = loc_after_direction(my_loc, move[0])
            risk = trap_risks.get(next_loc, 0.0)
            if risk < max_egg_risk:
                safe_egg_moves.append(move)
        
        if safe_egg_moves:
            # Prefer corners (worth 3 eggs)
            corner_moves = []
            for move in safe_egg_moves:
                if my_loc in [(0,0), (0,7), (7,0), (7,7)]:
                    corner_moves.append(move)
            return corner_moves[0] if corner_moves else safe_egg_moves[0]
        
        # Move toward nearest uncollected egg square
        nearest_egg = self._find_nearest_egg_square(board, my_loc)
        if nearest_egg:
            path = self.pathfinder.astar_path(board, my_loc, nearest_egg, trap_risks)
            if path and len(path) > 0:
                # path is a list of Directions
                direction = path[0]
                plain_move = (direction, MoveType.PLAIN)
                if plain_move in valid_moves:
                    next_loc = loc_after_direction(my_loc, direction)
                    next_risk = trap_risks.get(next_loc, 0.0)
                    
                    # DYNAMIC RISK: Accept higher risk if we're behind or stuck
                    egg_diff = my_eggs - opp_eggs
                    if egg_diff < -3:  # Significantly behind - TAKE RISKS
                        max_risk = 0.40
                    elif egg_diff < 0:  # Behind - accept moderate risk
                        max_risk = 0.32
                    else:  # Ahead or tied - be cautious
                        max_risk = 0.25
                    
                    if next_risk < max_risk:
                        # ANTI-OSCILLATION: Avoid if this returns to recent location
                        if len(self.move_history) >= 2:
                            recent = list(self.move_history)[-4:]
                            if next_loc not in recent:
                                return plain_move
                        else:
                            return plain_move
        
        # Fallback: move toward nearest corner
        corners = [(0,0), (0,7), (7,0), (7,7)]
        valid_corners = [c for c in corners if (c[0]+c[1])%2 == self.my_parity]
        if valid_corners:
            nearest_corner = min(valid_corners, key=lambda c: abs(my_loc[0]-c[0]) + abs(my_loc[1]-c[1]))
            for move in valid_moves:
                if move[1] != MoveType.PLAIN:
                    continue
                next_loc = loc_after_direction(my_loc, move[0])
                curr_dist = abs(my_loc[0] - nearest_corner[0]) + abs(my_loc[1] - nearest_corner[1])
                next_dist = abs(next_loc[0] - nearest_corner[0]) + abs(next_loc[1] - nearest_corner[1])
                if next_dist < curr_dist and trap_risks.get(next_loc, 0.0) < 0.30:
                    return move
        
        # FINAL DESPERATION FALLBACK: Pick least risky move (don't freeze!)
        plain_moves = [m for m in valid_moves if m[1] == MoveType.PLAIN]
        if plain_moves:
            # Sort by risk and pick safest
            move_risks = []
            for move in plain_moves:
                next_loc = loc_after_direction(my_loc, move[0])
                risk = trap_risks.get(next_loc, 0.0)
                move_risks.append((risk, move))
            move_risks.sort()
            return move_risks[0][1]  # Return lowest risk move
        
        return None

    def _find_nearest_egg_square(self, board: Board, my_loc: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest uncollected egg square."""
        size = board.game_map.MAP_SIZE
        best = None
        best_dist = 999
        for x in range(size):
            for y in range(size):
                loc = (x, y)
                if (x + y) % 2 != self.my_parity:
                    continue
                if loc in board.eggs_player or loc in board.eggs_enemy:
                    continue
                if loc in board.turds_enemy or loc in board.turds_player:
                    continue
                dist = abs(my_loc[0] - x) + abs(my_loc[1] - y)
                if dist < best_dist:
                    best_dist = dist
                    best = loc
        return best