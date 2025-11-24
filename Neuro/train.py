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

# --- INSANE MODE (i7-12700H + 3050Ti + 64GB RAM) ---
# 10 Workers * 128 Games = 1280 Parallel Games
# This creates massive batches for the GPU to chew on.
NUM_WORKERS = 8
PARALLEL_GAMES_PER_WORKER = 128  
MCTS_SIMS = 15           
ITERATIONS = 1000
EPOCHS = 1
BATCH_SIZE = 256         

CACHED_MAP = GameMap()

def setup_game():
    board = Board(CACHED_MAP)
    row = random.randint(1, 6)
    p1_start = (0, row)
    p2_start = (7, row)
    board.chicken_player.start(p1_start, (p1_start[0]+p1_start[1])%2)
    board.chicken_enemy.start(p2_start, (p2_start[0]+p2_start[1])%2)
    return board

def worker_process(rank, queue, model_state_dict):
    try:
        # 1. Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = ZeroNet().to(device)
        net.load_state_dict(model_state_dict)
        net.eval()
        
        mcts = NeuroMCTS(net)
        boards = [setup_game() for _ in range(PARALLEL_GAMES_PER_WORKER)]
        
        from .neuro_mcts import MCTSNode
        roots = [MCTSNode(b, prior=1.0) for b in boards]
        mcts.process_batch(roots)
        
        histories = [[] for _ in range(PARALLEL_GAMES_PER_WORKER)]
        active_indices = list(range(PARALLEL_GAMES_PER_WORKER))
        completed_data = []
        
        # 2. Game Loop
        while active_indices:
            # MCTS Phase
            for _ in range(MCTS_SIMS):
                leaves = []
                for idx in active_indices:
                    leaf = mcts.step_select(roots[idx])
                    if leaf.board.is_game_over():
                        diff = leaf.board.chicken_player.get_eggs_laid() - leaf.board.chicken_enemy.get_eggs_laid()
                        val = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
                        mcts.step_backprop(leaf, val)
                    else:
                        leaves.append(leaf)
                if leaves: mcts.process_batch(leaves)
            
            # Play Phase
            finished_indices = []
            for idx in active_indices:
                root = roots[idx]
                counts = np.zeros(12)
                for k, v in root.children.items(): counts[k] = v.visits
                s = np.sum(counts)
                probs = counts / s if s > 0 else np.ones(12)/12
                
                histories[idx].append([mcts.encode(root.board), probs, 0])
                
                move_idx = np.random.choice(12, p=probs)
                d, t = decode_move_idx(move_idx)
                
                if move_idx in root.children:
                    roots[idx] = root.children[move_idx]
                    roots[idx].parent = None
                    if roots[idx].board is None:
                        roots[idx].board = root.board.forecast_move(d, t, check_ok=False)
                        if not roots[idx].board:
                            finished_indices.append(idx)
                            continue
                        mcts.process_batch([roots[idx]])
                else:
                    new_b = root.board.forecast_move(d, t, check_ok=False)
                    if not new_b:
                        finished_indices.append(idx)
                        continue
                    roots[idx] = MCTSNode(new_b, prior=1.0)
                    mcts.process_batch([roots[idx]])
                
                if roots[idx].board.is_game_over():
                    finished_indices.append(idx)
            
            # Finish
            for idx in finished_indices:
                active_indices.remove(idx)
                b = roots[idx].board
                p1, p2 = b.chicken_player.get_eggs_laid(), b.chicken_enemy.get_eggs_laid()
                winner = 1.0 if p1 > p2 else (-1.0 if p2 > p1 else 0.0)
                
                h = histories[idx]
                for i in range(len(h)):
                    h[i][2] = winner * ((-1) ** (len(h) - 1 - i))
                    completed_data.append(h[i])
        
        queue.put((completed_data, mcts.timers))
        
    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        traceback.print_exc()

def train():
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Workers: {NUM_WORKERS}")
    
    net = ZeroNet().to(device)
    path = os.path.join(os.path.dirname(__file__), "brain.pth")
    
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=device, weights_only=False)
            net.load_state_dict(state)
            print("Loaded Brain.")
        except: print("New Brain.")
    
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
        combined_timers = {'encode': 0.0, 'gpu': 0.0, 'expand': 0.0, 'select': 0.0}
        finished_workers = 0
        
        while finished_workers < NUM_WORKERS:
            if not queue.empty():
                data, worker_timers = queue.get()
                all_data.extend(data)
                for k, v in worker_timers.items():
                    combined_timers[k] += v
                finished_workers += 1
                print(f".", end='', flush=True)
            else:
                time.sleep(0.1)
                
        for p in processes:
            p.join()
            
        gen_time = time.time() - start_time
        
        # --- FULL DETAILED LOGGING ---
        total_time_measured = sum(combined_timers.values())
        if total_time_measured == 0: total_time_measured = 1.0
        
        print(f"\n  > Generated {len(all_data)} samples in {gen_time:.1f}s ({len(all_data)/gen_time:.0f} samples/s)")
        print(f"  > Work Distribution (Avg per worker):")
        print(f"    [GPU Inference]  {combined_timers['gpu']/NUM_WORKERS:.1f}s  ({combined_timers['gpu']/total_time_measured*100:.1f}%) -> Neural Net")
        print(f"    [CPU Logic]      {combined_timers['expand']/NUM_WORKERS:.1f}s  ({combined_timers['expand']/total_time_measured*100:.1f}%) -> Rules/Moves")
        print(f"    [CPU Tree Walk]  {combined_timers['select']/NUM_WORKERS:.1f}s  ({combined_timers['select']/total_time_measured*100:.1f}%) -> Python Loop")
        print(f"    [CPU Encoding]   {combined_timers['encode']/NUM_WORKERS:.1f}s  ({combined_timers['encode']/total_time_measured*100:.1f}%) -> Board->Tensor")
        
        # 3. Train Phase
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
            
            torch.save(net.state_dict(), path)
            
        if i % 5 == 0: 
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    train()