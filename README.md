# 🐔 Chicken Tournament AI: Competitive Game-Playing Autonomous Agents

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20RL-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Course](https://img.shields.io/badge/Georgia%20Tech-CS%203600%20Intro%20to%20AI-B3A369.svg)](https://www.gatech.edu/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A suite of high-performance, competitive autonomous game-playing agents engineered for the **Georgia Tech CS 3600 Chicken Tournament**. Features probabilistic hidden-trapdoor belief modeling (Bayesian inference), iterative deepening Minimax with Alpha-Beta pruning, Monte Carlo Tree Search (MCTS), and deep neural network policy/value evaluation trained on GPU clusters.

---

## 📌 Overview

The **Chicken Tournament** is a competitive multi-agent grid game where two autonomous agents maneuver on a dynamic map with incomplete information (fog-of-war, hidden trapdoors, and opponent obstruction). Agents compete to harvest resources, lay eggs, navigate hazardous terrain, and strategically trap or deny territory to the opposing agent under strict real-time decision clock constraints (≤ 1.0s/turn).

This repository contains our complete development lineage, from foundational baseline bots to our top-tier competitive tournament contenders: **Chimera**, **Magnus**, and **NewNewNeuro**.

```
                           ┌────────────────────────────────────────┐
                           │      Game Engine & Sensor Input        │
                           └──────────────────┬─────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │  Bayesian Belief State  │                       │  Dynamic Map / Barriers │
        │  - Trapdoor Probability │                       │  - Reachable Regions    │
        │  - Sensor Fusion Update │                       │  - Opponent Trapping    │
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

## 🧠 Contender Agents & Architecture

| Contender Agent | Core Paradigms & Algorithms | Key Strengths |
| :--- | :--- | :--- |
| **`Chimera`** | Hybrid State Machine + Iterative Deepening Minimax + Dynamic Risk Pruning | High win-rate tournament engine; phase-aware egg harvesting, opponent path-blocking, and aggressive clock management. |
| **`Magnus`** | Monte Carlo Tree Search (MCTS) + Bayesian Trapdoor Filtering + Killer Heuristics | Deep tactical tree rollouts; superior multi-step horizon lookahead and trap risk minimization. |
| **`NewNewNeuro` / `Neuro`** | Deep Reinforcement Learning (PyTorch `brain.pth`) + Neural MCTS Policy | Convolutional neural network trained on A100 GPU compute to predict board value heuristics from high-dimensional game states. |
| **`Cordelia`** | Anti-diagonal Barrier Placement + Heuristic Pruning | Defensive zoning specialist that isolates opponent territory using strategic turd barrier mechanics. |
| **`Oracle`** | Pure Bayesian Sensor Inference + Multi-Hypothesis Tracking | High-accuracy trapdoor likelihood maps derived from sparse sensor observations. |
| **`Farmer` / `Baker` / `Blitz`** | Lawnmower Exploration Tours + A* / BFS Pathfinding | Deterministic spatial sweeping algorithms ensuring complete board coverage with minimal backtracking. |

---

## 🔬 Key Technical Innovations

### 1. Probabilistic Trapdoor Belief Modeling (`trapdoor_belief.py`)
- Maps hidden hazard zones under partial observability using sensor fusion.
- Evaluates transition likelihoods upon agent movement and sensor pings, updating posterior probability matrices for every coordinate.
- Automatically isolates and blacklists confirmed trapdoors while dynamically weighting soft risk thresholds (e.g., pruning paths with $P(\text{trap}) \ge 0.18$).

### 2. Deep Learning & MCTS Integration (`Neuro/model.py`, `neuro_mcts.py`)
- Built using **PyTorch** with residual convolutional layers mapping the grid state to policy distributions and scalar position evaluations.
- Includes automated training pipelines (`train_a100.py`, `train_realistic.py`) optimized for self-play data generation and GPU acceleration.

### 3. Iterative Deepening Minimax with Alpha-Beta Pruning
- Dynamic depth search capable of scaling between 8–14 ply within the 1-second timeout budget.
- Enhanced with **Transposition Tables (TT)**, **Killer Move Heuristics**, and **History Tables** to maximize branch pruning efficiency.

### 4. Phase-Aware Strategic Heuristics
- **Early Phase (< 12 turns)**: Prioritizes egg acquisition tours and rapid barrier breaching.
- **Mid Phase (12–28 turns)**: Adaptive pruning with territory obstruction, enemy mobility denial, and targeted barrier construction.
- **Endgame Phase (≥ 29 turns)**: Aggressive defense, score protection, and opponent entrapment.

---

## 📂 Repository Structure

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

## 🚀 Quick Start

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

## 👥 Credits & Academic Context

- **Course**: CS 3600 (Introduction to Artificial Intelligence) at the **Georgia Institute of Technology**.
- **Contributors**:
  - **Benjamin Yohros** ([@byohros6](https://github.com/byohros6)) - Architecture, Chimera/Magnus decision loop, Bayesian trapdoor belief modeling, and neural MCTS implementation.
  - **Arnav** ([@Arnav2610](https://github.com/Arnav2610)) - Collaboration on initial tournament environment integration and agent benchmarking.

*This repository serves as an academic and engineering portfolio showcase of game-theoretic AI, search algorithms, and reinforcement learning.*
