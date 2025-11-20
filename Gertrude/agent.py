import time
from typing import List, Tuple, Callable
import numpy as np
from game.board import Board
from game.enums import Direction, MoveType
from .trapdoor_belief import TrapdoorBelief

class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.tracker = TrapdoorBelief()
        self.time_per_turn = 0.5 

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        # --- TALK: Report Status ---
        my_loc = board.chicken_player.get_location()
        print(f"\n[Gertrude] I am at {my_loc}. Eggs: {board.chicken_player.get_eggs_laid()}")
        print(f"[Gertrude] Sensors: {sensor_data}")
        
        # 1. Update Beliefs
        self.tracker.update(my_loc, sensor_data)
        
        # Check danger level
        risk = self.tracker.get_prob_at(my_loc)
        if risk > 0.10:
            print(f"[Gertrude] WARN: Standing on possible trapdoor! Risk: {risk:.2f}")

        start_time = time.time()
        best_move = None
        best_score = float('-inf')
        
        # 2. Iterative Deepening Search
        depth = 1
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            print("[Gertrude] I have no moves! I am blocked.")
            return (Direction.UP, MoveType.PLAIN)

        best_move = valid_moves[0] 

        try:
            while True:
                if time.time() - start_time > self.time_per_turn:
                    break
                
                current_score, current_move = self.minimax(board, depth, float('-inf'), float('inf'), True, start_time)
                
                if current_move is None: # Timeout during search
                    break
                
                best_score = current_score
                best_move = current_move
                depth += 1
                
                if depth > 20: 
                    break
                    
        except Exception as e:
            print(f"[Gertrude] ERROR: {e}")

        # --- TALK: Report Decision ---
        time_taken = time.time() - start_time
        print(f"[Gertrude] Thinking complete in {time_taken:.3f}s.")
        print(f"[Gertrude] Reached Depth: {depth-1}")
        print(f"[Gertrude] Strategy: Playing {best_move} with Score {best_score}")

        return best_move

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, maximizing_player: bool, start_time: float):
        if time.time() - start_time > self.time_per_turn:
            return 0, None

        if depth == 0 or board.is_game_over():
            return self.evaluate(board), None

        valid_moves = board.get_valid_moves(enemy=not maximizing_player)
        
        if not valid_moves:
            return self.evaluate(board), None

        # Optimization: Sort moves to prune faster (Eggs > Turds > Plain)
        valid_moves.sort(key=lambda m: 0 if m[1] == MoveType.EGG else (1 if m[1] == MoveType.TURD else 2))

        best_move = valid_moves[0]

        if maximizing_player:
            max_eval = float('-inf')
            for direction, move_type in valid_moves:
                next_board = board.forecast_move(direction, move_type, check_ok=False)
                if next_board is None: continue

                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, False, start_time)
                
                if time.time() - start_time > self.time_per_turn:
                    return 0, None

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (direction, move_type)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move

        else: # Minimizing Player
            min_eval = float('inf')
            for direction, move_type in valid_moves:
                next_board = board.forecast_move(direction, move_type, check_ok=False)
                if next_board is None: continue

                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, True, start_time)
                
                if time.time() - start_time > self.time_per_turn:
                    return 0, None

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (direction, move_type)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, board: Board):
        # 1. Score Difference (Weighted HEAVILY)
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        # 1 egg = 1000 points. This is the most important stat.
        score = (my_eggs - enemy_eggs) * 1000 

        # 2. Trapdoor Safety
        my_loc = board.chicken_player.get_location()
        risk = self.tracker.get_prob_at(my_loc)
        if risk > 0.25: score -= 50000
        else: score -= risk * 100

        # 3. Mobility / Blocking
        my_moves = len(board.get_valid_moves(enemy=False))
        enemy_moves = len(board.get_valid_moves(enemy=True))
        score += (my_moves - enemy_moves) * 20
        if enemy_moves == 0: score += 100000 

        # 4. SMART Corner Bias
        # Only target corners where WE can lay eggs!
        # Player A (Even) -> (0,0), (7,7), etc.
        # Player B (Odd)  -> (0,7), (7,0), etc.
        
        my_parity = board.chicken_player.even_chicken # 0 or 1
        all_corners = [(0,0), (0,7), (7,0), (7,7)]
        
        # Filter for only VALID corners (matching our parity)
        valid_corners = [c for c in all_corners if (c[0] + c[1]) % 2 == my_parity]

        if my_loc in valid_corners:
            # We are in a GOOD corner.
            score += 3000 
        elif my_loc in all_corners:
            # We are in a BAD corner (wrong parity). Get out!
            score -= 500 
        else:
            # Move towards a GOOD corner
            if valid_corners:
                dists = [abs(my_loc[0]-cx) + abs(my_loc[1]-cy) for cx, cy in valid_corners]
                score -= min(dists) * 50 
            
            # Bonus for laying an egg on the way
            if board.can_lay_egg_at_loc(my_loc): score += 200

        return score