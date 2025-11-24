import os
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import random
import time
import sys
import traceback
import gc
from game.board import Board
from game.game_map import GameMap
from .model import ZeroNet
from .neuro_mcts import NeuroMCTS, decode_move_idx

# --- REALISTIC TRAINING CONFIG ---
NUM_WORKERS = 60                
PARALLEL_GAMES_PER_WORKER = 64 
MCTS_SIMS = 20
ITERATIONS = 1000
EPOCHS = 1
BATCH_SIZE = 4096               

MY_BRAIN = "brain_real.pth"     
V100_BRAIN = "brain.pth"        

CACHED_MAP = GameMap()

def generate_traps():
    """Generates 2 traps (1 white, 1 black) weighted towards center."""
    traps = []
    # Weights for rings 0, 1, 2, 3 (Center)
    # Coords for rings:
    # Ring 3 (Center): (3,3), (3,4), (4,3), (4,4) -> Weight 2
    # ...
    # Simplification: Rejection sampling based on weights
    
    for parity in [0, 1]:
        while True:
            r, c = random.randint(0, 7), random.randint(0, 7)
            if (r+c)%2 != parity: continue
            
            dist_edge = min(r, 7-r, c, 7-c)
            weight = 0
            if dist_edge >= 3: weight = 2
            elif dist_edge == 2: weight = 1
            
            # Roll dice to accept this trap
            if random.random() < (weight / 2.0): # 2 is max weight
                traps.append((r,c))
                break
                
    return traps

def setup_game():
    board = Board(CACHED_MAP)
    # A starts on Left/Right Edge (Col 0 or 7), not corner (Row 1-6)
    row = random.randint(1, 6)
    col_a = random.choice([0, 7])
    
    # B mirrors A
    col_b = 7 if col_a == 0 else 0
    
    p1_start = (col_a, row)
    p2_start = (col_b, row)
    
    board.chicken_player.start(p1_start, (p1_start[0]+p1_start[1])%2)
    board.chicken_enemy.start(p2_start, (p2_start[0]+p2_start[1])%2)
    
    # Inject Hidden Traps
    board.hidden_traps = generate_traps()
    
    # Create Perfect Belief Maps for training (Cheat Sheet)
    # We feed the network the ACTUAL trap locations so it learns to fear them.
    board.p1_belief = np.zeros((8,8))
    board.p2_belief = np.zeros((8,8))
    for tx, ty in board.hidden_traps:
        board.p1_belief[tx, ty] = 1.0
        board.p2_belief[tx, ty] = 1.0
        
    return board

def worker_process(rank, queue, model_state_dict):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(1)
        
        net = ZeroNet().to(device)
        net.load_state_dict(model_state_dict)
        net.eval()
        
        mcts = NeuroMCTS(net)
        boards = [setup_game() for _ in range(PARALLEL_GAMES_PER_WORKER)]
        
        from .neuro_mcts import MCTSNode
        roots = [MCTSNode(b, prior=1.0) for b in boards]
        # Initial expansion uses P1's belief
        mcts.process_batch(roots, belief=None) # encode will pull from board.p1_belief
        
        histories = [[] for _ in range(PARALLEL_GAMES_PER_WORKER)]
        active_indices = list(range(PARALLEL_GAMES_PER_WORKER))
        completed_data = []
        
        while active_indices:
            # MCTS Phase
            for _ in range(MCTS_SIMS):
                leaves = []
                for idx in active_indices:
                    # Pass current player's belief
                    # If is_as_turn, use p1_belief. Else p2.
                    b = roots[idx].board
                    belief = b.p1_belief if b.is_as_turn else b.p2_belief
                    mcts.current_belief = belief
                    
                    leaf = mcts.step_select(roots[idx])
                    if leaf.board.is_game_over():
                        me = leaf.board.chicken_player.get_eggs_laid()
                        en = leaf.board.chicken_enemy.get_eggs_laid()
                        val = 1.0 if me > en else (-1.0 if en > me else 0.0)
                        mcts.step_backprop(leaf, val)
                    else:
                        leaves.append(leaf)
                if leaves: mcts.process_batch(leaves)
            
            # Play Phase
            finished_indices = []
            for idx in active_indices:
                root = roots[idx]
                b = root.board
                belief = b.p1_belief if b.is_as_turn else b.p2_belief
                
                counts = np.zeros(12)
                for k, v in root.children.items(): counts[k] = v.visits
                s = np.sum(counts)
                probs = counts / s if s > 0 else np.ones(12)/12
                
                # Save with belief
                histories[idx].append([mcts.encode(root.board, belief), probs, 0])
                
                move_idx = np.random.choice(12, p=probs)
                d, t = decode_move_idx(move_idx)
                
                # --- EXECUTE MOVE WITH PHYSICS ---
                # 1. Forecast (Standard logic)
                new_b = b.forecast_move(d, t, check_ok=False)
                
                if not new_b:
                    # Invalid move -> Loss
                    winner = -1.0 # Current player loses
                    # (Simplified handling: Just mark finish)
                    finished_indices.append(idx)
                    continue

                # 2. Trap Physics
                loc = new_b.chicken_player.get_location() # New loc of mover
                if loc in b.hidden_traps:
                    # PUNISHMENT
                    # Enemy gets +4
                    new_b.chicken_enemy.eggs_laid += 4
                    
                    # Teleport to spawn
                    spawn = new_b.chicken_player.get_spawn()
                    
                    # Check if spawn is blocked
                    spawn_blocked = False
                    if spawn == new_b.chicken_enemy.get_location(): spawn_blocked = True
                    if spawn in new_b.eggs_player or spawn in new_b.eggs_enemy: spawn_blocked = True
                    # (Simplified block check)
                    
                    if spawn_blocked:
                        # Game Over, Penalty
                        # "Enemy given 4 eggs... game terminated" (Already gave 4 above)
                        new_b.winner = "TRAP_KILL" # Marker
                        finished_indices.append(idx) # Will resolve score later
                    else:
                        # Teleport
                        new_b.chicken_player.loc = spawn
                        
                # 3. Pass State
                new_b.hidden_traps = b.hidden_traps
                new_b.p1_belief = b.p1_belief
                new_b.p2_belief = b.p2_belief
                
                new_b.reverse_perspective()
                
                # Tree Reuse
                if move_idx in root.children:
                    roots[idx] = root.children[move_idx]
                    roots[idx].parent = None
                    roots[idx].board = new_b
                else:
                    roots[idx] = MCTSNode(new_b, prior=1.0)
                    
                mcts.process_batch([roots[idx]]) # Expand new root

                if new_b.is_game_over():
                    finished_indices.append(idx)
            
            # Finish
            for idx in finished_indices:
                active_indices.remove(idx)
                b = roots[idx].board
                
                # Calculate score
                p1_score = b.chicken_player.get_eggs_laid()
                p2_score = b.chicken_enemy.get_eggs_laid()
                
                # Flip back if we are looking at P2's perspective
                if not b.is_as_turn:
                    p1_score, p2_score = p2_score, p1_score
                
                win_val = 1.0 if p1_score > p2_score else (-1.0 if p2_score > p1_score else 0.0)
                
                h = histories[idx]
                for i in range(len(h)):
                    h[i][2] = win_val * ((-1) ** (len(h) - 1 - i))
                    completed_data.append(h[i])
        
        queue.put((completed_data, mcts.timers))
        
    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        traceback.print_exc()

def train():
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Workers: {NUM_WORKERS}")
    print(f"REALISTIC MODE. Saving to: {MY_BRAIN}")
    
    net = ZeroNet().to(device)
    
    base_path = os.path.dirname(__file__)
    my_path = os.path.join(base_path, MY_BRAIN)
    v100_path = os.path.join(base_path, V100_BRAIN)
    
    # Load previous brains if available to kickstart
    if os.path.exists(v100_path):
        try:
            state = torch.load(v100_path, map_location=device, weights_only=False)
            net.load_state_dict(state)
            print(f"Loaded baseline from {V100_BRAIN}.")
        except: pass
        
    net.share_memory()
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    scaler = torch.amp.GradScaler('cuda')

    for i in range(ITERATIONS):
        print(f"--- Iteration {i} ---")
        start_time = time.time()
        
        queue = mp.Queue()
        processes = []
        cpu_weights = {k: v.cpu() for k, v in net.state_dict().items()}
        
        print(f"  > Spawning {NUM_WORKERS} workers...", end='', flush=True)
        
        for rank in range(NUM_WORKERS):
            p = mp.Process(target=worker_process, args=(rank, queue, cpu_weights))
            p.start()
            processes.append(p)
            
        all_data = []
        finished_workers = 0
        while finished_workers < NUM_WORKERS:
            if not queue.empty():
                data, _ = queue.get()
                all_data.extend(data)
                finished_workers += 1
                print(f".", end='', flush=True)
            else:
                time.sleep(0.1)
        for p in processes: p.join()
        
        gen_time = time.time() - start_time
        print(f"\n  > Generated {len(all_data)} samples in {gen_time:.1f}s")
        
        if len(all_data) > 0:
            net.train()
            states = np.stack([x[0] for x in all_data])
            probs = np.stack([x[1] for x in all_data])
            values = np.stack([x[2] for x in all_data]).reshape(-1, 1)
            
            indices = np.arange(len(states))
            for _ in range(EPOCHS):
                np.random.shuffle(indices)
                for j in range(0, len(states), BATCH_SIZE):
                    batch_idx = indices[j:j+BATCH_SIZE]
                    s_t = torch.from_numpy(states[batch_idx]).float().to(device)
                    p_t = torch.from_numpy(probs[batch_idx]).float().to(device)
                    v_t = torch.from_numpy(values[batch_idx]).float().to(device)
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        pred_log_p, pred_v = net(s_t)
                        loss = -torch.mean(torch.sum(p_t * pred_log_p, 1)) + \
                                torch.nn.functional.mse_loss(pred_v, v_t)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            
            torch.save(net.state_dict(), my_path)
            
        if i % 5 == 0: 
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    train()