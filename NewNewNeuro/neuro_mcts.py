import math
import numpy as np
import torch
import time
from game.enums import Direction, MoveType

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
        self.timers = {'encode': 0.0, 'gpu': 0.0, 'expand': 0.0, 'select': 0.0}
        # CACHE: Stores (policy_numpy, value_float) for board hash
        self.cache = {}
        
    def search_setup(self, root_board, belief_map=None):
        """Initializes a search tree and clears cache for the new turn."""
        self.cache.clear() # Reset cache for new game turn (time has changed)
        
        root = MCTSNode(root_board, prior=1.0)
        self.current_belief = belief_map
        self._expand_sequential(root)
        return root

    def search_batch(self, root, batch_size=1):
        """CPU Optimized: Batch size 1 is usually best."""
        nodes_to_expand = []
        for _ in range(batch_size):
            node = self.step_select(root)
            if not node.board.is_game_over():
                nodes_to_expand.append(node)
            else:
                self._backprop_terminal(node)
        
        if nodes_to_expand:
            self.process_batch(nodes_to_expand)

    def get_probs(self, root):
        counts = np.zeros(12)
        for idx, child in root.children.items():
            counts[idx] = child.visits
        if np.sum(counts) == 0: return np.ones(12) / 12.0
        return counts / np.sum(counts)

    def step_select(self, root):
        t0 = time.perf_counter()
        node = root
        while node.children: 
            best_score = -float('inf')
            best_child = None
            sqrt_parent = math.sqrt(node.visits)
            for child in node.children.values():
                if child.visits == 0:
                    best_child = child
                    break
                score = (child.value_sum / child.visits) + (child.prior * sqrt_parent / (1 + child.visits))
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            
            if node.board is None:
                d, t = node.move_from_parent
                node.board = node.parent.board.forecast_move(d, t, check_ok=False)
                # Training Fix: Copy attributes if they exist
                if node.board and hasattr(node.parent.board, 'hidden_traps'):
                    node.board.hidden_traps = node.parent.board.hidden_traps
                    node.board.p1_belief = node.parent.board.p1_belief
                    node.board.p2_belief = node.parent.board.p2_belief
                
                if node.board: node.board.reverse_perspective()
                if node.board and node.board.is_game_over(): break
        
        self.timers['select'] += (time.perf_counter() - t0)
        return node

    def step_backprop(self, node, value):
        while node:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def _backprop_terminal(self, node):
        me = node.board.chicken_player.get_eggs_laid()
        en = node.board.chicken_enemy.get_eggs_laid()
        val = 1.0 if me > en else (-1.0 if en > me else 0.0)
        self.step_backprop(node, val)

    def encode(self, board, belief=None):
        return self._encode(board, belief)

    def process_batch(self, nodes, belief=None):
        if not nodes: return
        
        t0 = time.perf_counter()
        b_map = belief if belief is not None else getattr(self, 'current_belief', None)
        
        # 1. ENCODING & CACHE LOOKUP
        uncached_nodes = []
        uncached_states = []
        cached_results = []
        
        for node in nodes:
            state = self._encode(node.board, b_map)
            # Create a hashable key from the numpy array bytes
            state_key = state.tobytes()
            
            if state_key in self.cache:
                # HIT! Use stored result
                cached_results.append((node, self.cache[state_key]))
            else:
                # MISS! Queue for inference
                uncached_nodes.append((node, state_key))
                uncached_states.append(state)
                
        self.timers['encode'] += (time.perf_counter() - t0)
        
        # 2. INFERENCE (Only for misses)
        if uncached_states:
            t1 = time.perf_counter()
            
            # Convert to tensor
            batch_t = torch.tensor(np.stack(uncached_states), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Remove autocast for CPU (it can slow things down or crash on some CPUs)
                if self.device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        policy_logits, values = self.net(batch_t)
                else:
                    policy_logits, values = self.net(batch_t)
            
            self.timers['gpu'] += (time.perf_counter() - t1)
            
            policy_probs = torch.exp(policy_logits).cpu().numpy()
            values = values.cpu().numpy().flatten()
            
            # Store in Cache and Expand
            t2 = time.perf_counter()
            for i, (node, key) in enumerate(uncached_nodes):
                p, v = policy_probs[i], values[i]
                self.cache[key] = (p, v) # Save for later
                self._expand_node_with_probs(node, p, v)
            self.timers['expand'] += (time.perf_counter() - t2)
            
        # 3. EXPAND CACHED (Fast path)
        t3 = time.perf_counter()
        for node, (p, v) in cached_results:
            self._expand_node_with_probs(node, p, v)
        self.timers['expand'] += (time.perf_counter() - t3)

    def _expand_sequential(self, node):
        # Reuse the batched logic since it now handles caching
        self.process_batch([node])

    def _expand_node_with_probs(self, node, priors, value):
        valid_moves = node.board.get_valid_moves()
        sum_valid = 0.0
        for d, t in valid_moves:
            idx = encode_move_idx(d, t)
            p = priors[idx]
            child = MCTSNode(board=None, parent=node, prior=p, move_from_parent=(d,t))
            node.children[idx] = child
            sum_valid += p
        if sum_valid > 0:
            scale = 1.0 / sum_valid
            for child in node.children.values():
                child.prior *= scale
        self.step_backprop(node, value)

    def _encode(self, board, belief_map=None):
        state = np.zeros((8, 8, 8), dtype=np.float32)
        me, en = board.chicken_player, board.chicken_enemy
        mx, my = me.get_location()
        ex, ey = en.get_location()
        state[0, mx, my] = 1
        state[1, ex, ey] = 1
        for (x, y) in board.eggs_player: state[2, x, y] = 1
        for (x, y) in board.eggs_enemy: state[3, x, y] = 1
        for (x, y) in board.turds_player: state[4, x, y] = 1
        for (x, y) in board.turds_enemy: state[5, x, y] = 1
        
        if belief_map is not None:
            state[6] = belief_map
        elif hasattr(board, 'p1_belief') and hasattr(board, 'is_as_turn'):
             state[6] = board.p1_belief if board.is_as_turn else board.p2_belief
        elif hasattr(board, 'hidden_traps'):
            for (tx, ty) in board.hidden_traps:
                state[6, tx, ty] = 1.0
        elif hasattr(board, 'found_trapdoors'):
             for (tx, ty) in board.found_trapdoors:
                state[6, tx, ty] = 1.0
             
        state[7, :, :] = board.turn_count / 80.0
        return state