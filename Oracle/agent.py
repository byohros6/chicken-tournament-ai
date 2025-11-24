import time
import random
import sys

try:
    from game.enums import Direction, MoveType
except ImportError:
    pass

from .trapdoor_belief import TrapdoorBelief
from .gamestate import GameState
from .heuristic import HeuristicEvaluator

class PlayerAgent:
    def __init__(self, game_state, time_left_fn):
        self.belief_engine = TrapdoorBelief(8, 8)
        self.time_buffer = 0.05 
        self.my_id = None
        self.evaluator = None

    def play(self, game_state, sensor_data, time_left_fn):
        my_chicken = game_state.chicken_player
        
        if self.my_id is None:
            if my_chicken.even_chicken == 0:
                self.my_id = 0 # Player A
            else:
                self.my_id = 1 # Player B
            self.evaluator = HeuristicEvaluator(self.my_id)

        current_pos = my_chicken.get_location()

        if sensor_data:
            self.belief_engine.update(current_pos, sensor_data)
            
        self.belief_engine.normalize_after_safe_move(*current_pos)

        root_state = GameState.from_engine_board(game_state, self.my_id)

        time_remaining = time_left_fn()
        turns_played = root_state.turn_count // 2
        turns_left = 40 - turns_played
        if turns_left < 1: turns_left = 1
        
        base_budget = time_remaining / turns_left
        
        if 10 < turns_played < 30:
            time_budget = base_budget * 1.5
        else:
            time_budget = base_budget * 0.8

        time_budget = min(time_budget, 8.0) 
        time_budget = max(time_budget, 0.2) 
        
        deadline = time.time() + time_budget - self.time_buffer

        best_move_str = None
        depth = 1
        self.nodes_visited = 0
        
        legal_moves = root_state.get_legal_actions(self.my_id)
        if not legal_moves:
            return (Direction.UP, MoveType.PLAIN)
        
        best_move_str = random.choice(legal_moves)

        try:
            while True:
                if time.time() > deadline:
                    break
                
                move, score = self.alpha_beta(
                    root_state, depth, float('-inf'), float('inf'), 
                    True, deadline
                )
                
                if move is not None:
                    best_move_str = move
                
                if score > 90000: break
                depth += 1
                if depth > 12: break 
                
        except TimeoutError:
            pass 

        return self._convert_to_enums(best_move_str)

    def alpha_beta(self, state, depth, alpha, beta, is_maximizing, deadline):
        if self.nodes_visited % 1000 == 0:
            if time.time() > deadline:
                raise TimeoutError()
        self.nodes_visited += 1

        if depth == 0 or state.is_game_over:
            return None, self.evaluator.evaluate(state, self.belief_engine)

        player_id = self.my_id if is_maximizing else (1 - self.my_id)
        moves = state.get_legal_actions(player_id)
        
        if not moves:
            return None, self.evaluator.evaluate(state, self.belief_engine)

        moves.sort(key=lambda m: 0 if m[1] == 'EGG' else (1 if m[1] == 'TURD' else 2))

        best_move = moves[0]

        if is_maximizing:
            max_eval = float('-inf')
            for move in moves:
                child = state.apply_action(player_id, move)
                _, eval_score = self.alpha_beta(child, depth - 1, alpha, beta, False, deadline)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                child = state.apply_action(player_id, move)
                _, eval_score = self.alpha_beta(child, depth - 1, alpha, beta, True, deadline)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return best_move, min_eval

    def _convert_to_enums(self, move_tuple):
        d_str, t_str = move_tuple
        
        if d_str == "UP": d_enum = Direction.UP
        elif d_str == "DOWN": d_enum = Direction.DOWN
        elif d_str == "LEFT": d_enum = Direction.LEFT
        elif d_str == "RIGHT": d_enum = Direction.RIGHT
        
        if t_str == "PLAIN": t_enum = MoveType.PLAIN
        elif t_str == "EGG": t_enum = MoveType.EGG
        elif t_str == "TURD": t_enum = MoveType.TURD
        
        return (d_enum, t_enum)

class TimeoutError(Exception):
    pass