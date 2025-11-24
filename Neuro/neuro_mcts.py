import math
import numpy as np
import torch
import time
from game.enums import Direction, MoveType

# Pre-computed constants
DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
TYPES = [MoveType.PLAIN, MoveType.EGG, MoveType.TURD]

def encode_move_idx(direction, m_type):
    d_idx = DIRECTIONS.index(direction)
    t_idx = TYPES.index(m_type)
    return d_idx * 3 + t_idx

def decode_move_idx(idx):
    return (DIRECTIONS[idx // 3], TYPES[idx % 3])

class MCTSNode:
    __slots__ = ['board', 'parent', 'children', 'visits', 'value_sum', 'prior', 'move_from_parent']
    
    def __init__(self, board, parent=None, prior=0.0, move_from_parent=None):
        self.board = board
        self.parent = parent
        self.children = {} 
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.move_from_parent = move_from_parent

class NeuroMCTS:
    def __init__(self, net):
        self.net = net
        self.device = next(net.parameters()).device
        # Initialize timers
        self.timers = {'encode': 0.0, 'gpu': 0.0, 'expand': 0.0, 'select': 0.0}
        
    def search(self, root_board, num_sims=50):
        """Sequential search (For Agent Play)."""
        root = MCTSNode(root_board, prior=1.0)
        self._expand_sequential(root)
        
        for _ in range(num_sims):
            node = self.step_select(root)
            if not node.board.is_game_over():
                self._expand_sequential(node)
            else:
                self._backprop_terminal(node)
                
        counts = np.zeros(12)
        for idx, child in root.children.items():
            counts[idx] = child.visits
            
        if np.sum(counts) == 0:
            return np.ones(12) / 12.0 # Safety fallback
            
        return counts / np.sum(counts)

    # --- PROFILING ENABLED METHODS ---

    def step_select(self, root):
        t0 = time.perf_counter() # START TIMER
        node = root
        
        while node.children: 
            best_score = -float('inf')
            best_child = None
            sqrt_parent = math.sqrt(node.visits)
            
            for child in node.children.values():
                if child.visits == 0:
                    best_child = child
                    break
                
                # UCT Formula
                score = (child.value_sum / child.visits) + \
                        (child.prior * sqrt_parent / (1 + child.visits))
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child
            
            # Lazy Expansion Trigger
            if node.board is None:
                d, t = node.move_from_parent
                # Forecast the move
                node.board = node.parent.board.forecast_move(d, t, check_ok=False)
                
                # --- CRITICAL PERSPECTIVE FIX ---
                # We must flip the board so the network always sees "Me" vs "Enemy"
                if node.board:
                    node.board.reverse_perspective()

                if node.board and node.board.is_game_over():
                    break
        
        self.timers['select'] += (time.perf_counter() - t0) # STOP TIMER
        return node

    def step_backprop(self, node, value):
        while node:
            node.visits += 1
            node.value_sum += value
            value = -value # Flip value for opponent
            node = node.parent

    def _backprop_terminal(self, node):
        # Since we flipped perspective, "chicken_player" is the one who just moved
        me = node.board.chicken_player.get_eggs_laid()
        en = node.board.chicken_enemy.get_eggs_laid()
        
        # If I (who just moved) have more eggs, I win (+1)
        val = 1.0 if me > en else (-1.0 if en > me else 0.0)
        
        self.step_backprop(node, val)

    def encode(self, board):
        return self._encode(board)

    def process_batch(self, nodes):
        if not nodes: return
        
        # 1. Encode
        t0 = time.perf_counter()
        batch_states = np.stack([self._encode(n.board) for n in nodes])
        batch_t = torch.tensor(batch_states, dtype=torch.float32).to(self.device)
        self.timers['encode'] += (time.perf_counter() - t0)
        
        # 2. GPU Inference
        t1 = time.perf_counter()
        with torch.no_grad():
            # Use float16 for speed if available (Laptop/A100)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                policy_logits, values = self.net(batch_t)
            
        self.timers['gpu'] += (time.perf_counter() - t1)
        
        policy_probs = torch.exp(policy_logits).float().cpu().numpy()
        values = values.float().cpu().numpy().flatten()
        
        # 3. Expand
        t2 = time.perf_counter()
        for i, node in enumerate(nodes):
            self._expand_node_with_probs(node, policy_probs[i], values[i])
        self.timers['expand'] += (time.perf_counter() - t2)

    def _expand_sequential(self, node):
        """Helper for single-mode."""
        state = self._encode(node.board)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p, v = self.net(state_t)
        self._expand_node_with_probs(node, torch.exp(p).cpu().numpy()[0], v.item())

    def _expand_node_with_probs(self, node, priors, value):
        # get_valid_moves works on "chicken_player" (Current Turn)
        valid_moves = node.board.get_valid_moves()
        sum_valid = 0.0
        
        for d, t in valid_moves:
            idx = encode_move_idx(d, t)
            p = priors[idx]
            
            # Create "Ghost Node" (Board is None until visited)
            child = MCTSNode(board=None, parent=node, prior=p, move_from_parent=(d,t))
            node.children[idx] = child
            sum_valid += p
        
        # Normalize priors
        if sum_valid > 0:
            scale = 1.0 / sum_valid
            for child in node.children.values():
                child.prior *= scale
                
        self.step_backprop(node, value)

    def _encode(self, board):
        state = np.zeros((8, 8, 8), dtype=np.float32)
        # Since board is always flipped to "My Perspective", player is always Me
        me, en = board.chicken_player, board.chicken_enemy
        mx, my = me.get_location()
        ex, ey = en.get_location()
        
        state[0, mx, my] = 1
        state[1, ex, ey] = 1
        for (x, y) in board.eggs_player: state[2, x, y] = 1
        for (x, y) in board.eggs_enemy: state[3, x, y] = 1
        for (x, y) in board.turds_player: state[4, x, y] = 1
        for (x, y) in board.turds_enemy: state[5, x, y] = 1
        state[7, :, :] = board.turn_count / 80.0
        return state