import time
import numpy as np
import heapq
from collections import deque
from typing import List, Tuple, Callable, Optional
from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction

# Ensure this import works for your file structure
from .belief import TrapdoorBelief

class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.tracker = TrapdoorBelief()
        self.time_limit = 0.45
        self.max_depth = 25
        
        # --- ADVANCED MEMORY ---
        self.history = deque(maxlen=20) # Track long-term movement
        self.current_target = None      # The specific tile we are hunting
        self.stuck_counter = 0          # Are we dancing in circles?
        
        # Static Goal: The Anti-Diagonal (Perfect defense line)
        self.wall_targets = set()
        for x in range(8):
            for y in range(8):
                if 6 <= (x + y) <= 8: # Diagonal band
                    self.wall_targets.add((x,y))

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        # 1. Update Beliefs
        my_loc = board.chicken_player.get_location()
        self.tracker.update(my_loc, sensor_data)
        self.history.append(my_loc)

        # 2. Detect Stuck Condition
        # If we have stayed in the same 3 squares for the last 10 turns, PANIC.
        if len(self.history) >= 10 and len(set(list(self.history)[-10:])) <= 3:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # 3. PATHFINDING (The Brain Upgrade)
        # Before we search, pick a SMART target using BFS.
        self.current_target = self.select_strategic_target(board)
        
        # Log intent (Optional, helps debugging)
        # print(f"Targeting: {self.current_target} | Stuck: {self.stuck_counter}")

        start_time = time.time()
        valid_moves = board.get_valid_moves()
        if not valid_moves: return (Direction.UP, MoveType.PLAIN)
        
        # Sort Moves
        valid_moves.sort(key=lambda m: self.score_move_ordering(m))
        best_move = valid_moves[0]

        # 4. Iterative Deepening Minimax
        try:
            for depth in range(1, self.max_depth + 1):
                if time.time() - start_time > self.time_limit: break
                
                score, move = self.minimax(board, depth, float('-inf'), float('inf'), True, start_time)
                
                if move:
                    best_move = move
                    if score > 90000: break
        except: pass

        return best_move

    def select_strategic_target(self, board: Board) -> Optional[Tuple[int, int]]:
        """
        Uses BFS to find the nearest REACHABLE, VALUABLE square.
        Ignores corners if they are blocked off.
        """
        my_loc = board.chicken_player.get_location()
        parity = board.chicken_player.even_chicken
        
        # If we are stuck, pick a random safe square far away
        if self.stuck_counter > 0:
            # Find any valid empty square far from here
            best_escape = None
            max_dist = 0
            for x in range(8):
                for y in range(8):
                    if self.is_safe_and_empty(board, (x,y), parity):
                        d = abs(my_loc[0]-x) + abs(my_loc[1]-y)
                        if d > max_dist:
                            max_dist = d
                            best_escape = (x,y)
            return best_escape

        # BFS Initialization
        queue = deque([(my_loc, 0)]) # (loc, distance)
        visited = {my_loc}
        
        best_target = None
        
        # Search for targets in order of proximity
        while queue:
            curr, dist = queue.popleft()
            
            # A. Is this a good target?
            # 1. Corners (High Priority)
            if curr in [(0,0), (0,7), (7,0), (7,7)]:
                if self.is_safe_and_empty(board, curr, parity):
                    return curr # Found reachable corner!
            
            # 2. Empty Valid Squares (Medium Priority)
            # We store the first one we find, but keep looking for corners briefly
            if best_target is None and self.is_safe_and_empty(board, curr, parity):
                best_target = curr
                # Don't return immediately; check a bit deeper for corners
                
            # Limit BFS depth to keep it fast (e.g., look 10 steps away)
            if dist > 12: break

            # B. Expand Neighbors
            for d in Direction:
                try:
                    next_loc = loc_after_direction(curr, d)
                    if board.is_valid_cell(next_loc) and next_loc not in visited:
                        # Can we walk here?
                        if not board.is_cell_blocked(next_loc):
                            visited.add(next_loc)
                            queue.append((next_loc, dist + 1))
                except: pass
        
        return best_target

    def is_safe_and_empty(self, board, loc, parity):
        """Helper to validate a target tile."""
        # Parity
        if (loc[0] + loc[1]) % 2 != parity: return False
        # Occupied?
        if loc in board.eggs_player or loc in board.eggs_enemy: return False
        if loc in board.turds_player or loc in board.turds_enemy: return False
        # Trap?
        if self.tracker.get_prob_at(loc) > 0.20: return False
        return True

    def score_move_ordering(self, move_tuple):
        # Heuristic for sorting moves
        move_type = move_tuple[1]
        if move_type == MoveType.EGG: return 0
        if move_type == MoveType.TURD: return 1
        return 2

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, is_max: bool, start_time: float):
        if time.time() - start_time > self.time_limit: return 0, None
        if depth == 0 or board.is_game_over(): return self.evaluate(board), None

        moves = board.get_valid_moves(enemy=not is_max)
        if not moves: return (-1000000 if is_max else 1000000), None

        if is_max:
            moves.sort(key=lambda m: self.score_move_ordering(m))

        best_move = moves[0]

        if is_max:
            max_eval = float('-inf')
            for d, t in moves:
                next_b = board.forecast_move(d, t, check_ok=False)
                if not next_b: continue
                score, _ = self.minimax(next_b, depth-1, alpha, beta, False, start_time)
                if score > max_eval:
                    max_eval = score
                    best_move = (d, t)
                alpha = max(alpha, score)
                if beta <= alpha: break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for d, t in moves:
                next_b = board.forecast_move(d, t, check_ok=False)
                if not next_b: continue
                score, _ = self.minimax(next_b, depth-1, alpha, beta, True, start_time)
                if score < min_eval:
                    min_eval = score
                    best_move = (d, t)
                beta = min(beta, score)
                if beta <= alpha: break
            return min_eval, best_move

    def evaluate(self, board: Board):
        # 1. SCORE
        my_eggs = board.chicken_player.get_eggs_laid()
        en_eggs = board.chicken_enemy.get_eggs_laid()
        score = (my_eggs - en_eggs) * 1000

        my_loc = board.chicken_player.get_location()

        # 2. SAFETY
        risk = self.tracker.get_prob_at(my_loc)
        if risk > 0.20: return -50000 

        # 3. TARGETING (Pathfinding Guided)
        # If we have a valid BFS target, pull towards it
        if self.current_target:
            # Use Manhattan distance for speed in leaf nodes
            dist = abs(my_loc[0]-self.current_target[0]) + abs(my_loc[1]-self.current_target[1])
            score -= dist * 50
            
            # Bonus for BEING there (The Capture)
            if dist == 0 and board.can_lay_egg_at_loc(my_loc):
                score += 500
        else:
            # No path found? We are trapped.
            # Switch to pure survival/aggression
            pass

        # 4. DEFENSIVE WALL (Smart Turds)
        # Only reward turds on the Anti-Diagonal (x+y ~ 7)
        # This prevents "Top Wall" clustering.
        for t in board.turds_player:
            if 6 <= (t[0] + t[1]) <= 8:
                score += 300
        
        # 5. AGGRESSION
        en_moves = len(board.get_valid_moves(enemy=True))
        score -= en_moves * 5

        return score