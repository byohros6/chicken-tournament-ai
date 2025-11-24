import os
import torch
import numpy as np
import time
from game.enums import Direction, MoveType
from .model import ZeroNet
from .neuro_mcts import NeuroMCTS, decode_move_idx, encode_move_idx
from .belief import TrapdoorBelief

class PlayerAgent:
    def __init__(self, board, time_left):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ZeroNet().to(self.device)
        self.brain_ready = False
        
        self.belief = TrapdoorBelief()
        self.location_history = [] 
        
        path = os.path.join(os.path.dirname(__file__), "brain.pth")
        if os.path.exists(path):
            try:
                # weights_only=False allows loading legacy/custom structures safely
                state = torch.load(path, map_location=self.device, weights_only=False)
                self.net.load_state_dict(state)
                self.net.eval()
                # Warmup
                dummy = torch.zeros((1, 8, 8, 8), device=self.device)
                self.net(dummy)
                self.mcts = NeuroMCTS(self.net)
                self.brain_ready = True
                print("NEURO: Grandmaster Brain Loaded.")
            except Exception as e:
                print(f"NEURO: Load failed! {e}")
        
    def play(self, board, sensor_data, time_left):
        try:
            valid_moves = board.get_valid_moves()
            if not valid_moves: return (Direction.UP, MoveType.PLAIN)

            my_loc = board.chicken_player.get_location()
            self.belief.update(my_loc, sensor_data)
            if hasattr(board, 'found_trapdoors'):
                self.belief.set_confirmed_traps(board.found_trapdoors)
            
            if not self.brain_ready:
                return valid_moves[0]
            
            # Dynamic Time Management
            turns_remaining = board.turns_left_player
            my_time = time_left()
            usable_time = my_time - 2.0
            
            if usable_time <= 0.5:
                thinking_time = 0.0
            else:
                base_budget = usable_time / (turns_remaining + 2)
                thinking_time = min(base_budget, 2.5)
            
            start_t = time.time()
            root = self.mcts.search_setup(board, belief_map=self.belief.probs)
            
            if thinking_time > 0.05:
                while time.time() - start_t < thinking_time:
                    self.mcts.search_batch(root, batch_size=1)
            
            probs = self.mcts.get_probs(root)
            
            # --- DICTATORIAL SELECTION LOGIC ---
            best_move = valid_moves[0]
            best_score = -999999.0
            
            # Define Corners
            CORNERS = [(0,0), (0,7), (7,0), (7,7)]
            
            for d, t in valid_moves:
                idx = encode_move_idx(d, t)
                move_prob = probs[idx]
                
                # Calculate Target
                dx, dy = 0, 0
                if d == Direction.UP: dy = -1
                elif d == Direction.DOWN: dy = 1
                elif d == Direction.LEFT: dx = -1
                elif d == Direction.RIGHT: dx = 1
                
                target_r = my_loc[0] + dy
                target_c = my_loc[1] + dx
                target_loc = (target_r, target_c)
                
                # Penalties
                penalty = 0.0
                
                # 1. CORNER EVICTION (MISSING IN YOUR FILE)
                # If we visited this corner recently, BAN IT.
                if target_loc in CORNERS and target_loc in self.location_history:
                    penalty = 10.0 
                
                # 2. IMMEDIATE BACKTRACK BAN (MISSING IN YOUR FILE)
                # If A -> B -> A, BAN IT (unless trapped).
                elif len(self.location_history) >= 1 and target_loc == self.location_history[-1]:
                    penalty = 2.0
                
                # 3. STANDARD HISTORY PENALTY
                elif target_loc in self.location_history:
                    penalty = 0.5
                
                score = move_prob - penalty
                
                # 4. FORCE EGG (WEAK IN YOUR FILE)
                if t == MoveType.PLAIN:
                    can_lay = False
                    for vd, vt in valid_moves:
                        if vd == d and vt == MoveType.EGG:
                            can_lay = True
                            break
                    if can_lay:
                        score -= 1.0 # Was 0.1, now 1.0 to guarantee EGG
                
                if score > best_score:
                    best_score = score
                    best_move = (d, t)
            
            self.location_history.append(my_loc)
            if len(self.location_history) > 12: 
                self.location_history.pop(0)
            
            return best_move

        except Exception as e:
            print(f"NEURO CRASHED: {e}")
            valid_moves = board.get_valid_moves()
            return valid_moves[0] if valid_moves else (Direction.UP, MoveType.PLAIN)