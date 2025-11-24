import os
import torch
import numpy as np
from game.enums import Direction, MoveType
from .model import ZeroNet
from .neuro_mcts import NeuroMCTS, decode_move_idx

class PlayerAgent:
    def __init__(self, board, time_left):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ZeroNet().to(self.device)
        self.brain_ready = False
        
        path = os.path.join(os.path.dirname(__file__), "brain.pth")
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=self.device, weights_only=False)
                self.net.load_state_dict(state)
                self.net.eval()
                self.mcts = NeuroMCTS(self.net)
                self.brain_ready = True
                print("NEURO: Brain Loaded.")
            except:
                print("NEURO: Brain load failed.")
        
    def play(self, board, sensor_data, time_left):
        if not self.brain_ready:
            moves = board.get_valid_moves()
            return moves[0] if moves else (Direction.UP, MoveType.PLAIN)
            
        # 800 Sims = Grandmaster Calculation
        probs = self.mcts.search(board, num_sims=800)
        move_idx = np.argmax(probs)
        return decode_move_idx(move_idx)