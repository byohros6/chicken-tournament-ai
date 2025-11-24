import time
import math
from game.enums import MoveType, Direction
from .graph_utils import GraphManager 

class SearchAgent:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.transposition_table = {}
        self.start_time = 0
        self.time_limit = 0
        self.nodes_explored = 0
        
        # --- STRATEGIC WEIGHTS ---
        self.W_EGG = 100.0           # High value for scoring
        self.W_MOBILITY = 10.0       # Good value for freedom of movement
        self.W_TURD_HELD = 50.0      # AMMO CONSERVATION: Don't waste turds on nothing!
        self.W_REPEAT_PENALTY = 200.0 # Massive penalty for wiggling back and forth
        self.W_EGG_POTENTIAL = 10.0   # Bonus for standing on a square you can score on

    def get_best_move(self, board, oracle, time_left_fn, visit_counts=None):
        if visit_counts is None: visit_counts = {}
        
        self.start_time = time.time()
        total_time_left = time_left_fn()
        
        # Dynamic Time Budgeting
        move_budget = min(2.5, total_time_left / 10.0)
        if total_time_left < 2.0: move_budget = 0.2

        self.time_limit = self.start_time + move_budget
        self.nodes_explored = 0
        
        best_move = None
        current_depth = 1
        MAX_DEPTH = 30 # Safety Cap
        
        # Optimization: if only 1 move is valid, take it instantly
        valid_moves = board.get_valid_moves()
        if len(valid_moves) == 1:
            return valid_moves[0]

        while current_depth <= MAX_DEPTH:
            try:
                score, move = self._minimax(board, current_depth, -math.inf, math.inf, True, oracle, visit_counts)
                if move is not None:
                    best_move = move
                
                if score > 8000: break # Found a winning line
                current_depth += 1
                
            except TimeoutError:
                break
            except RecursionError:
                break
                
        if best_move is None:
            return valid_moves[0] if valid_moves else (Direction.UP, MoveType.PLAIN)
            
        return best_move

    def _minimax(self, board, depth, alpha, beta, maximizing_player, oracle, visit_counts):
        if self.nodes_explored % 50 == 0:
            if time.time() > self.time_limit: raise TimeoutError
        self.nodes_explored += 1

        if depth == 0:
            return self._evaluate(board, oracle, visit_counts), None

        board_key = self._hash_board(board)
        if board_key in self.transposition_table:
            entry = self.transposition_table[board_key]
            if entry['depth'] >= depth:
                return entry['score'], entry['move']

        valid_moves = board.get_valid_moves(enemy=not maximizing_player)
        if not valid_moves:
            return self._evaluate(board, oracle, visit_counts), None

        # Move Ordering: Check Eggs first, then Turds
        valid_moves.sort(key=lambda m: (m[1] == MoveType.EGG, m[1] == MoveType.TURD), reverse=True)
        
        best_move = valid_moves[0]

        if maximizing_player:
            max_eval = -math.inf
            for move in valid_moves:
                next_board = board.forecast_move(move[0], move[1], check_ok=False)
                if next_board is None: continue
                
                eval_score, _ = self._minimax(next_board, depth - 1, alpha, beta, False, oracle, visit_counts)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            
            self.transposition_table[board_key] = {'score': max_eval, 'move': best_move, 'depth': depth}
            return max_eval, best_move

        else:
            min_eval = math.inf
            for move in valid_moves:
                next_board = board.forecast_move(move[0], move[1], check_ok=False)
                if next_board is None: continue

                eval_score, _ = self._minimax(next_board, depth - 1, alpha, beta, True, oracle, visit_counts)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha: break
            
            self.transposition_table[board_key] = {'score': min_eval, 'move': best_move, 'depth': depth}
            return min_eval, best_move

    def _evaluate(self, board, oracle, visit_counts):
        # 1. Scoring
        my_eggs = board.chicken_player.get_eggs_laid()
        opp_eggs = board.chicken_enemy.get_eggs_laid()
        score = (my_eggs - opp_eggs) * self.W_EGG
        
        # 2. Ammo Conservation (Don't waste turds)
        my_turds = board.chicken_player.get_turds_left()
        score += my_turds * self.W_TURD_HELD
        
        my_loc = board.chicken_player.get_location()
        
        # 3. Risk Management
        risk = oracle.get_risk(my_loc)
        if risk > 0.0: score -= (risk * 2000.0)

        # 4. History Penalty (Anti-Wiggle)
        visits = visit_counts.get(my_loc, 0)
        if visits > 0:
            score -= (visits * self.W_REPEAT_PENALTY)

        # 5. Tactical Bonuses
        # Egg Potential: Am I standing on a square I can score on?
        is_my_parity = ((my_loc[0] + my_loc[1]) % 2) == board.chicken_player.even_chicken
        if is_my_parity and my_loc not in board.eggs_player:
            score += self.W_EGG_POTENTIAL

        # Mobility (Fast check)
        try:
            my_moves = len(board.get_valid_moves(enemy=False))
            opp_moves = len(board.get_valid_moves(enemy=True))
            score += (my_moves - opp_moves) * self.W_MOBILITY
            
            # Kill Bonus: If opponent is trapped, huge score
            if opp_moves <= 2: score += 500 
        except:
            pass

        return score
    
    def _hash_board(self, board):
        p_loc = board.chicken_player.get_location()
        e_loc = board.chicken_enemy.get_location()
        p_eggs = frozenset(board.eggs_player)
        e_eggs = frozenset(board.eggs_enemy)
        p_turds = frozenset(board.turds_player)
        e_turds = frozenset(board.turds_enemy)
        return hash((p_loc, e_loc, p_eggs, e_eggs, p_turds, e_turds))