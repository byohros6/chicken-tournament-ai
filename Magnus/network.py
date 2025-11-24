"""
Neural network for policy and value prediction.
Simplified version inspired by AlphaZero.
"""
import numpy as np
from typing import Tuple, List
import pickle
import os


class ChickenNet:
    """
    Neural network with policy and value heads.
    For now, uses simple heuristics - can be replaced with trained network later.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.trained = False
        
        # Try to load trained model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy (move probabilities) and value (position evaluation).
        
        Args:
            board_state: Encoded board state (see encode_board)
        
        Returns:
            (policy, value) where:
                - policy: array of move probabilities [Direction * MoveType]
                - value: scalar value in [-1, 1]
        """
        if self.trained:
            return self._nn_predict(board_state)
        else:
            return self._heuristic_predict(board_state)
    
    def _heuristic_predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Heuristic-based prediction (fallback when no trained model).
        """
        # Extract features from board state
        # board_state shape: (channels, 8, 8)
        
        # Get my position and eggs
        my_pos = self._extract_position(board_state[0])  # My position channel
        my_eggs = np.sum(board_state[2])  # My eggs channel
        enemy_eggs = np.sum(board_state[3])  # Enemy eggs channel
        
        # Value: simple egg difference
        egg_diff = my_eggs - enemy_eggs
        value = np.tanh(egg_diff / 10.0)  # Normalize to [-1, 1]
        
        # Policy: uniform over valid directions (will be filtered by MCTS)
        # 4 directions * 3 move types = 12 possible moves
        policy = np.ones(12) / 12.0
        
        return policy, float(value)
    
    def _nn_predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Neural network prediction (placeholder for future implementation).
        """
        # TODO: Implement with PyTorch/JAX
        # For now, fall back to heuristics
        return self._heuristic_predict(board_state)
    
    def _extract_position(self, position_channel: np.ndarray) -> Tuple[int, int]:
        """Extract (row, col) from one-hot position channel."""
        pos = np.argwhere(position_channel > 0)
        if len(pos) > 0:
            return tuple(pos[0])
        return (0, 0)
    
    def load_model(self, path: str) -> bool:
        """Load trained model from file."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            # TODO: Load actual PyTorch/JAX model
            self.trained = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.trained = False
            return False
    
    def save_model(self, path: str) -> bool:
        """Save trained model to file."""
        try:
            # TODO: Save actual PyTorch/JAX model
            with open(path, 'wb') as f:
                pickle.dump({}, f)
            return True
        except Exception as e:
            print(f"Failed to save model: {e}")
            return False


def encode_board(board) -> np.ndarray:
    """
    Encode board state as tensor for neural network.
    
    Channels:
    0: My chicken position (one-hot)
    1: Enemy chicken position (one-hot)
    2: My eggs (3 for corners, 1 for regular)
    3: Enemy eggs (3 for corners, 1 for regular)
    4: Turds (1 for present, 0 for empty)
    5: Trapdoor probabilities (0-1)
    6: Turn number (normalized to 0-1)
    7: Time remaining (normalized to 0-1)
    
    Returns:
        np.ndarray of shape (8, 8, 8)
    """
    state = np.zeros((8, 8, 8), dtype=np.float32)
    
    # Channel 0: My position
    my_pos = board.chicken_player.get_pos()
    state[0, my_pos[0], my_pos[1]] = 1.0
    
    # Channel 1: Enemy position
    enemy_pos = board.chicken_enemy.get_pos()
    state[1, enemy_pos[0], enemy_pos[1]] = 1.0
    
    # Channel 2 & 3: Eggs
    for r in range(8):
        for c in range(8):
            square = board.get_square(r, c)
            
            # My eggs
            if square.get_eggs_p1() > 0:
                is_corner = (r in [0, 7] and c in [0, 7])
                state[2, r, c] = 3.0 if is_corner else 1.0
            
            # Enemy eggs
            if square.get_eggs_p2() > 0:
                is_corner = (r in [0, 7] and c in [0, 7])
                state[3, r, c] = 3.0 if is_corner else 1.0
    
    # Channel 4: Turds
    for r in range(8):
        for c in range(8):
            if board.get_square(r, c).has_turd():
                state[4, r, c] = 1.0
    
    # Channel 5: Trapdoor probabilities (placeholder - needs belief tracking)
    # This would be filled in by the agent
    
    # Channel 6: Turn number
    turn = board.get_num_turns()
    state[6, :, :] = turn / 40.0  # Normalize to [0, 1]
    
    # Channel 7: Time remaining (placeholder)
    # This would be filled in by the agent
    
    return state


def decode_move_index(index: int) -> Tuple:
    """
    Decode move index to (Direction, MoveType).
    Indices 0-11 map to 4 directions × 3 move types.
    """
    from game.enums import Direction, MoveType
    
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    move_types = [MoveType.PLAIN, MoveType.EGG, MoveType.TURD]
    
    dir_idx = index // 3
    move_idx = index % 3
    
    return (directions[dir_idx], move_types[move_idx])


def encode_move_index(direction, move_type) -> int:
    """
    Encode (Direction, MoveType) to move index.
    """
    from game.enums import Direction, MoveType
    
    dir_map = {
        Direction.UP: 0,
        Direction.DOWN: 1,
        Direction.LEFT: 2,
        Direction.RIGHT: 3
    }
    
    move_map = {
        MoveType.PLAIN: 0,
        MoveType.EGG: 1,
        MoveType.TURD: 2
    }
    
    return dir_map[direction] * 3 + move_map[move_type]
