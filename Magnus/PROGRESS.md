# Magnus - Strategic Chicken AI

## Test Results

### Magnus vs Yolanda (Random Agent)
- **Result: Magnus wins 9-2**
- Time: 19.77 seconds
- Magnus dominated with superior egg collection and strategic positioning

### Magnus vs Gertrude (Minimax with Trapdoor Belief)
- **Result: Magnus wins 6-2**
- Time: 50.69 seconds  
- Gertrude got stuck in oscillation loops
- Magnus showed better time management and strategic egg collection

## Current Implementation Features

✅ **Bayesian Trapdoor Inference**
- Log probabilities for numerical stability
- Proper decay (0.03) to avoid overconfidence
- Confirmed trapdoor tracking

✅ **A* Pathfinding**
- Optimal paths considering trapdoor risk
- Egg tour computation with corner prioritization
- 2-opt optimization support

✅ **Strategic Planning**
- Game phase detection (early/mid/endgame)
- Smart turd placement based on opponent threats
- High-value square identification

✅ **Minimax Search**
- Alpha-beta pruning
- Iterative deepening (depth 1-12)
- Transposition table for caching
- Killer move heuristic
- History heuristic for move ordering

✅ **Time Management**
- Conservative budgeting (60% of average time per move)
- Safety margin (50ms reserved)
- Adaptive search depth based on game phase

## Evaluation Function Weights

1. **Egg differential**: 5000 per egg (most important)
2. **Mobility**: 300 per additional move
3. **Trapdoor risk**: -2000 for current position risk
4. **Corner control**: Up to 1000 bonus for corners
5. **Distance to egg squares**: -20 per manhattan distance
6. **Turd advantage**: 50 per turd
7. **Center control**: -15 for distance from center
8. **Enemy distance**: -5 (slight aggression)

## Next Steps: Advanced ML Techniques

### Phase 2: AlphaZero-Style Improvements

Based on lectures/4_02-AlphaZero.pdf:

1. **Monte Carlo Tree Search (MCTS)**
   - Replace pure minimax with MCTS for better exploration
   - UCT (Upper Confidence bounds for Trees) selection
   - Backup values from rollouts

2. **Neural Network Policy + Value**
   - Policy network: P(move|state) - which moves are good
   - Value network: V(state) - how good is this position
   - Train on self-play games

3. **Self-Play Training Loop**
   - Generate games with current best agent
   - Use MCTS + neural net to play both sides
   - Save (state, policy, outcome) tuples
   - Train network on this data
   - Iterate

4. **Data Augmentation**
   - Board symmetries (rotations, reflections)
   - Generate more training data from each game

### Phase 3: Additional Enhancements

1. **Opening Book**
   - Pre-compute strong openings from self-play
   - Instant moves for first 3-5 turns

2. **Endgame Tablebase**
   - Solve all positions with <10 eggs remaining
   - Perfect play in endgame

3. **Ensemble Methods**
   - Combine multiple strategies
   - Vote on best move

4. **Reinforcement Learning**
   - Q-learning for position evaluation
   - Reward shaping for better training

## Implementation Strategy

**Short term (next 2 hours):**
- Test Magnus vs Cordelia
- Fine-tune evaluation weights based on weaknesses
- Add opening book for fast starts

**Medium term (1-2 days):**
- Implement basic MCTS
- Collect self-play data (1000 games)
- Train simple neural network

**Long term (before Dec 2):**
- Full AlphaZero pipeline
- Large-scale self-play
- Ensemble of best agents
