# Chicken Tournament AI: Competitive Game-Playing Autonomous Agents

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20RL-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Course](https://img.shields.io/badge/Georgia%20Tech-CS%203600%20Intro%20to%20AI-B3A369.svg)](https://www.gatech.edu/)
[![Tournament](https://img.shields.io/badge/ByteFight-Tournament%20Engine-black.svg)](https://bytefight.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A suite of competitive autonomous game-playing agents engineered for the **Georgia Tech CS 3600 Artificial Intelligence Tournament (ByteFight)**. The project implements Bayesian hidden hazard belief estimation, iterative deepening Minimax with Alpha-Beta pruning, Monte Carlo Tree Search (MCTS), and deep neural network evaluation trained on GPU clusters.

---

## Tournament Rules and Game Mechanics

The tournament is an adversarial, partially observable game played on an 8x8 board between two agents (Player A / White and Player B / Black):

- **Turn Limit & Clock**: 40 turns per chicken. Each agent operates under a strict **6-minute cumulative chess clock** for the entire match. Running out of time results in an immediate forfeit.
- **Scoring & Egg Mechanics**:
  - **Plain Step**: Moves to an adjacent cardinal square.
  - **Egg Step**: Drops an egg on the current square and exits. White lays on even-parity squares ($(i+j) \pmod 2 = 0$), Black on odd-parity squares ($(i+j) \pmod 2 = 1$).
  - **Corner Bonus**: Laying an egg on any of the four corner squares awards **3 eggs total** (1 base + 2 bonus).
  - The player with the highest egg count at turn 40 wins.
- **Territory Denial & Turd Barriers**:
  - Each agent can drop up to **5 turds** during the game (valid only when not directly adjacent to the opponent).
  - An agent can traverse its own eggs and turds, but **cannot enter any square containing an opponent's egg or turd**.
  - **Zoning Effect**: An agent **cannot enter any square that shares an edge with an opponent's turd**, allowing aggressive territorial cut-offs and corridor traps.
  - If a chicken has no valid legal moves, the opponent is awarded a 5-egg bonus and the game terminates.
- **Trapdoors & Spatial Hazard Physics**:
  - Two hidden trapdoors (one on a white square, one on a black square), weighted toward the center of the board.
  - Stepping on a trapdoor teleports the chicken back to its starting square and awards the opponent a **4-egg bonus**.
  - Entering any square generates noisy acoustic and vibration sensor observations for both trapdoor engines:
    - **Edge Adjacent**: 50% probability of hearing, 30% probability of feeling.
    - **Diagonal Adjacent**: 25% probability of hearing, 15% probability of feeling.
    - **Distance-2 Edge**: 10% probability of hearing, 0% probability of feeling.

---

## System Architecture

```
                           ┌────────────────────────────────────────┐
                           │      Game Engine & Sensor Signals      │
                           │   (heard_white, felt_white, ...)       │
                           └──────────────────┬─────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │  Bayesian Belief Filter │                       │  Territory & Mobility   │
        │  - Sensor Likelihoods   │                       │  - Reachable Components │
        │  - Posterior Risk Grid  │                       │  - Turd Exclusion Zones │
        │  - Center-Weight Prior  │                       │  - Corner Target Queue  │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │      Hybrid Decision Architecture       │
                         │   - Strategic Planner (Game Phase)      │
                         │   - Iterative Deepening Alpha-Beta      │
                         │   - Monte Carlo Tree Search (MCTS)      │
                         │   - Deep Neural Value Network (PyTorch) │
                         └────────────────────┬────────────────────┘
                                              │
                                              ▼
                                   Optimal Tournament Move
```

---

## Contender Agents

| Agent | Core Paradigms & Algorithms | Key Strengths |
| :--- | :--- | :--- |
| **`Chimera`** | Hybrid State Machine + Iterative Deepening Minimax + Dynamic Risk Pruning | Top tournament contender; phase-aware egg harvesting, opponent corridor trapping, and aggressive clock management. |
| **`Magnus`** | Monte Carlo Tree Search (MCTS) + Bayesian Trapdoor Filtering + Killer Heuristics | Deep tactical tree rollouts; multi-step lookahead and trap risk minimization. |
| **`NewNewNeuro` / `Neuro`** | Deep Reinforcement Learning (PyTorch `brain.pth`) + Neural MCTS Policy | Convolutional neural network trained on cloud GPU compute (A100) to evaluate board value heuristics from raw board state representations. |
| **`Cordelia`** | Anti-diagonal Barrier Placement + Heuristic Pruning | Defensive zoning specialist that isolates opponent territory using strategic turd barrier placements. |
| **`Oracle`** | Bayesian Sensor Inference + Multi-Hypothesis Tracking | High-accuracy trapdoor likelihood maps derived from sparse sensor observations. |
| **`Farmer` / `Baker` / `Blitz`** | Lawnmower Exploration Tours + A* / BFS Pathfinding | Deterministic spatial sweeping algorithms ensuring complete board coverage with minimal backtracking. |

---

## Technical Highlights

### 1. Bayesian Trapdoor Likelihood Estimation (`trapdoor_belief.py`)
Because stepping on a trapdoor inflicts a 4-egg deficit and teleports the agent backward, navigating around trapdoors is critical. The belief engine maintains posterior probability distributions over all 64 board squares:
- Initializes board priors using the official center-weighted distribution.
- Computes conditional sensor updates for acoustic and vibration signals:
  $$P(\text{hear} \mid d=1) = 0.50, \quad P(\text{feel} \mid d=1) = 0.30$$
  $$P(\text{hear} \mid \text{diag}) = 0.25, \quad P(\text{feel} \mid \text{diag}) = 0.15$$
  $$P(\text{hear} \mid d=2) = 0.10, \quad P(\text{feel} \mid d=2) = 0.00$$
- Hard-prunes candidate trapdoors based on dynamic threshold bands ($P(\text{trap}) \ge 0.18$), dynamically relaxing risk tolerance only during endgame deficits.

### 2. Deep Learning and MCTS Integration (`Neuro/model.py`, `neuro_mcts.py`)
- Engineered with **PyTorch** using convolutional residual layers mapping board representations (parity, eggs, turds, hazard belief) to policy distributions and scalar position evaluations.
- Includes training scripts (`train_a100.py`, `train_realistic.py`) designed for parallel game self-play data generation and GPU acceleration.

### 3. Iterative Deepening Minimax with Alpha-Beta Pruning
- Dynamically scales search depth between 8 and 14 ply based on remaining time on the 6-minute chess clock.
- Incorporates **Transposition Tables (TT)**, **Killer Move Heuristics**, and **History Tables** to optimize search efficiency.

### 4. Phase-Aware Heuristic Controller
- **Early Phase (< 12 turns)**: Prioritizes egg sweeps, corner tile bonuses (3 eggs each), and rapid barrier breakthroughs.
- **Mid Phase (12–28 turns)**: Adaptive pruning with territory obstruction, enemy mobility denial, and targeted turd placements.
- **Endgame Phase (≥ 29 turns)**: Defensive zoning, score preservation, and opponent entrapment.

---

## Repository Structure

```
chicken-tournament-ai/
├── Chimera/               # State-of-the-art hybrid tournament contender
│   ├── agent.py           # Core agent logic and decision loop
│   ├── pathfinding.py     # BFS and A* pathing with trap hazard avoidance
│   ├── strategy.py        # Phase controller and strategic planner
│   ├── trapdoor_belief.py # Bayesian sensor fusion and risk grid
│   └── PLAN.md            # Architectural design blueprint
├── Magnus/                # MCTS search and lookahead agent
├── NewNewNeuro/           # Neural network policy engine (A100 trained)
│   ├── model.py           # PyTorch neural network architecture
│   ├── neuro_mcts.py      # MCTS guided by neural value estimates
│   ├── train_a100.py      # Cloud GPU distributed training script
│   └── brain.pth          # Exported PyTorch model weights
├── Oracle/                # Sensor reasoning and probabilistic heuristic agent
├── Farmer/                # Lawnmower sweeping tour and crash memory
├── Blitz/                 # High-tempo heuristic agent
├── Cordelia/              # Barrier containment specialist
└── Gertrude/              # BFS-driven safe-tile pathfinder
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- PyTorch (for neural agents)

```bash
# Clone the repository
git clone https://github.com/byohros6/chicken-tournament-ai.git
cd chicken-tournament-ai

# (Optional) Install PyTorch for Neuro agents
pip install torch
```

---

## Academic Context and Contributions

- **Course**: CS 3600 (Introduction to Artificial Intelligence) at the **Georgia Institute of Technology**.
- **Platform**: ByteFight Tournament.
- **Contributors**:
  - **Benjamin Yohros** ([@byohros6](https://github.com/byohros6)) - Architecture, Chimera and Magnus decision loops, Bayesian trapdoor belief modeling, and neural MCTS implementation.
  - **Arnav** ([@Arnav2610](https://github.com/Arnav2610)) - Collaboration on tournament environment integration and agent benchmarking.

*This repository serves as an academic and engineering portfolio showcase of game-theoretic AI, search algorithms, and reinforcement learning.*
