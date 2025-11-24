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

# --- REALISTIC TRAINING ---
NUM_WORKERS = 30                # Tuned for A100 (32 Cores)
PARALLEL_GAMES_PER_WORKER = 128 
MCTS_SIMS = 20
ITERATIONS = 1000
EPOCHS = 1
BATCH_SIZE = 4096               

# New Save File
MY_BRAIN = "brain_real.pth"     
# Use previous A100 brain as baseline (Smartest one you have)
BASELINE_BRAIN = "brain_a100.pth"        

CACHED_MAP = GameMap()

def generate_traps():
    """Generates 1 White and 1 Black trap, weighted by distance from edge."""
    traps = []
    for parity in [0, 1]:
        while True:
            r, c = random.randint(0, 7), random.randint(0, 7)
            if (r+c)%2 != parity: continue
            
            dist_edge = min(r, 7-r, c, 7-c)
            weight = 0
            if dist_edge >= 3: weight = 2
            elif dist_edge == 2: weight = 1
            
            # Simple rejection sampling
            if random.random() < (weight / 2.0):
                traps.append((r,c))
                break
    return traps

def setup_game():
    board = Board(CACHED_MAP)
    
    # Rule: Chicken A starts on White Square on Left(0) or Right(7) Edge
    col_a = random.choice([0, 7])
    
    # If Col 0 (Even), Row must be Even (2,4,6) to be White
    # If Col 7 (Odd), Row must be Odd (1,3,5) to be White
    if col_a == 0:
        valid_rows = [2, 4, 6]
    else:
        valid_rows = [1, 3, 5]
    row = random.choice(valid_rows)
    
    # Rule: Chicken B mirrors A
    col_b = 7 if col_a == 0 else 0
    
    p1_start = (col_a, row)
    p2_start = (col_b, row)
    
    # Rule: A is White(0), B is Black(1)
    board.chicken_player.start(p1_start, 0) 
    board.chicken_enemy.start(p2_start, 1)
    
    # Inject Real Traps
    board.hidden_traps = generate_traps()
    
    # Perfect Belief for Training (The "Cheat Sheet")
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
        # Process P1 (White) initially
        mcts.process_batch(roots) 
        
        histories = [[] for _ in range(PARALLEL_GAMES_PER_WORKER)]
        active_indices = list(range(PARALLEL_GAMES_PER_WORKER))
        completed_data = []
        
        while active_indices:
            # MCTS
            for _ in range(MCTS_SIMS):
                leaves = []
                for idx in active_indices:
                    # Switch belief based on whose turn it is
                    b = roots[idx].board
                    belief = b.p1_belief if b.is_as_turn else b.p2_belief
                    mcts.current_belief = belief # For _encode
                    
                    leaf = mcts.step_select(roots[idx])
                    if leaf.board.is_game_over():
                        me = leaf.board.chicken_player.get_eggs_laid()
                        en = leaf.board.chicken_enemy.get_eggs_laid()
                        val = 1.0 if me > en else (-1.0 if en > me else 0.0)
                        mcts.step_backprop(leaf, val)
                    else:
                        leaves.append(leaf)
                if leaves: mcts.process_batch(leaves)
            
            # Play
            finished_indices = []
            for idx in active_indices:
                root = roots[idx]
                b = root.board
                belief = b.p1_belief if b.is_as_turn else b.p2_belief
                
                counts = np.zeros(12)
                for k, v in root.children.items(): counts[k] = v.visits
                s = np.sum(counts)
                probs = counts / s if s > 0 else np.ones(12)/12
                
                histories[idx].append([mcts.encode(root.board, belief), probs, 0])
                
                move_idx = np.random.choice(12, p=probs)
                d, t = decode_move_idx(move_idx)
                
                # --- PHYSICS ENGINE ---
                # 1. Validate Move Logic
                if move_idx in root.children:
                    roots[idx] = root.children[move_idx]
                    roots[idx].parent = None
                    # Lazy Expansion check
                    if roots[idx].board is None:
                        # Apply move (Standard)
                        new_b = b.forecast_move(d, t, check_ok=False)
                        
                        # TRAP CHECK (The Punishment)
                        if new_b:
                            loc = new_b.chicken_player.get_location()
                            if loc in b.hidden_traps:
                                # Penalty + Teleport
                                new_b.chicken_enemy.eggs_laid += 4
                                new_b.chicken_player.loc = new_b.chicken_player.get_spawn()
                                
                                # Check spawn block
                                spawn = new_b.chicken_player.get_spawn()
                                if spawn == new_b.chicken_enemy.get_location() or \
                                   spawn in new_b.eggs_player or spawn in new_b.eggs_enemy:
                                   # Instant Game Over (Loss)
                                   new_b.winner = "TRAP_DEATH" 
                        
                        roots[idx].board = new_b
                        if not new_b or new_b.winner == "TRAP_DEATH":
                             finished_indices.append(idx)
                             continue
                             
                        # Pass state & flip
                        new_b.hidden_traps = b.hidden_traps
                        new_b.p1_belief = b.p1_belief
                        new_b.p2_belief = b.p2_belief
                        new_b.reverse_perspective()
                        
                        mcts.process_batch([roots[idx]])
                else:
                    # Fallback (rare)
                    new_b = b.forecast_move(d, t, check_ok=False)
                    if not new_b: 
                        finished_indices.append(idx)
                        continue
                    # Trap Check
                    loc = new_b.chicken_player.get_location()
                    if loc in b.hidden_traps:
                        new_b.chicken_enemy.eggs_laid += 4
                        new_b.chicken_player.loc = new_b.chicken_player.get_spawn()
                        
                    new_b.hidden_traps = b.hidden_traps
                    new_b.p1_belief = b.p1_belief
                    new_b.p2_belief = b.p2_belief
                    new_b.reverse_perspective()
                    
                    roots[idx] = MCTSNode(new_b, prior=1.0)
                    mcts.process_batch([roots[idx]])

                if roots[idx].board.is_game_over():
                    finished_indices.append(idx)
            
            # Finish
            for idx in finished_indices:
                active_indices.remove(idx)
                b = roots[idx].board
                
                # Score Logic
                if b and b.winner == "TRAP_DEATH":
                    # Player who just moved (b.chicken_player) died. They lose (-1).
                    win_val = -1.0
                elif b:
                    p1 = b.chicken_player.get_eggs_laid()
                    p2 = b.chicken_enemy.get_eggs_laid()
                    # Flip back if perspective is swapped
                    if not b.is_as_turn: p1, p2 = p2, p1
                    win_val = 1.0 if p1 > p2 else (-1.0 if p2 > p1 else 0.0)
                else:
                    win_val = -1.0 # Invalid move
                
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
    baseline_path = os.path.join(base_path, BASELINE_BRAIN)
    
    # Resume or Fork
    loaded = False
    if os.path.exists(my_path):
        try:
            state = torch.load(my_path, map_location=device, weights_only=False)
            net.load_state_dict(state)
            print(f"Resumed from {MY_BRAIN}.")
            loaded = True
        except: pass
        
    if not loaded and os.path.exists(baseline_path):
        try:
            print(f"Forking from {BASELINE_BRAIN}...")
            state = torch.load(baseline_path, map_location=device, weights_only=False)
            net.load_state_dict(state)
            print("Success!")
            loaded = True
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
        if i % 5 == 0: gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    train()