# Chimera Overhaul Blueprint

## 1. Objectives & Success Criteria
- **Board dominance**: Maintain ≥65% win rate versus Magnus, Cordelia, Gertrude, Oracle in 20-game batches (10 as Player A, 10 as Player B).
- **Time safety**: Never exceed 90% of allotted clock; average decision time ≤1.0s midgame.
- **Trap discipline**: Hard-prune trap squares with risk ≥0.18 unless trailing by ≥3 eggs or ≤6 turns remain.
- **Mobility lock**: Penalize states where the enemy mobility advantage >4 reachable squares and reward trapping scenarios.

## 2. High-Level Architecture
| Component | Source Inspiration | Implementation Notes |
|-----------|--------------------|----------------------|
| Trap belief + PathFinder | Magnus | Keep existing modules; expose cached risk grid each turn. |
| Phase controller | Cordelia + Magnus | Reuse `StrategicPlanner.get_game_phase`, but enhance with barrier progress + egg differential triggers. |
| Target selector | Magnus tour + Gertrude BFS | Two-stage pipeline: tour target → BFS fallback → high-value heuristic. |
| Search loop | Magnus core | Iterative deepening alpha-beta with TT + killer/history tables; add stuck detection + jitter. |
| Heuristic | Hybrid | Blend Cordelia urgency, Oracle mobility, Magnus trap penalty, Gertrude target pull, endgame denial logic. |
| Turd policy | Cordelia barrier + Gertrude diagonal | Extend `StrategicPlanner` helpers to evaluate anti-diagonal walls and proximity denial. |

## 3. Detailed Feature Map
1. **Phase-Aware Strategy**
   - Early (<12): prioritize egg rush + barrier breach; allow single forced barrier build if opportunity is free.
   - Mid (12–28): enable adaptive pruning (Cordelia thresholds) and start barrier/turd operations with enemy threat awareness.
   - End (>=29): switch heuristic weights to survival (Oracle mobility diff) + denial; allow risky trap steps when leading by ≥3 eggs.

2. **Targeting Pipeline**
   1. `egg_tour` (Magnus) remains primary; recompute when blocked, stale (≥8 turns), or eggs exhausted.
   2. If tour invalid: call new `self._bfs_target(board, trap_risks)` (Gertrude-style) to select reachable safe tile.
   3. If stuck cycle detected (≤3 unique squares over 8 turns): override with farthest safe tile to escape.

3. **Search Enhancements**
   - Maintain iterative deepening but cap depth dynamically (e.g., 10 early, 12 mid) based on time budget.
   - Adaptive pruning splits into three bands:
     - **Safe**: risk < dynamic threshold (base 0.15, +0.02 if behind, +0.04 when ≤6 turns).
     - **Neutral**: threshold ≤ risk < 0.4, only considered when <2 safe moves.
     - **Desperation**: risk ≥0.4, only if no alternatives and egg deficit ≥2.
   - Inject 5% random shuffle among top-scoring root moves when scores within 150.

4. **Heuristic Stack**
   - Base: egg differential × 1000.
   - Trap penalty: `risk * 12000` scaled by phase.
   - Mobility term: `((my_moves - enemy_moves) * 60)` plus Oracle-style BFS count for endgame (reward trapping enemy).
   - Target pull: `-dist_to_target * 120 + on_target_bonus 400`.
   - Barrier awareness: penalty for being on wrong side of detected wall; bonus for turds on anti-diagonal band (6 ≤ x+y ≤ 8).
   - Endgame denial: if leading, penalize proximity (<3) to enemy; if trailing, incentivize closing distance and blocking.

5. **Turd Logic**
   - Add `StrategicPlanner.get_barrier_plan(board)` returning next anti-diagonal tile if path is safe.
   - New helper `_should_punch_barrier(board)` to prioritize breaking enemy walls when stuck behind.
   - Lay turd when: (a) midgame threat level ≥3 and move blocks target, (b) anti-diagonal plan tile is safe, or (c) endgame lead + enemy adjacency to high-value square.

6. **State Tracking Utilities**
   - `self.recent_locs` deque (len 8) to detect loops.
   - `self.escape_target` to chase when stuck.
   - `self.time_usage_stats` to smooth per-phase budgets (optional).

## 4. Implementation Checklist
1. **State scaffolding** (agent init): add `recent_locs`, `escape_target`, `last_phase`, `mobility_cache`.
2. **Tour & target orchestration**: new helpers `_needs_tour_refresh`, `_select_primary_target`, `_select_escape_target`.
3. **Adaptive pruning update**: integrate dynamic risk thresholds + panic modes.
4. **Heuristic rewrite**: implement modular sub-scores for eggs, traps, mobility, target pull, barrier pressure, endgame denial.
5. **Turd planner hooks**: extend `StrategicPlanner` with barrier/turd helpers and call from agent when `MoveType.TURD` chosen.
6. **Loop escape + randomness**: detect loops and bias move ordering; add tie-break jitter at root.
7. **Testing harness**: scripts for `Chimera vs {Magnus,Cordelia,Gertrude,Oracle}` (10 games each) + regression log under `matches/`.

## 5. Risks & Mitigations
- **Over-expensive search**: keep real-time budget guard (Magnus logic) + early bailout when depth > 10 and branching high.
- **Barrier mis-detection**: log detection state and allow manual reset every 8 turns if enemy turds removed.
- **Trap aggression causing losses**: threshold scaling tied to egg deficit and turn count keeps suicide moves limited.

## 6. Next Steps
1. Wire new state + helper scaffolding into `Chimera/agent.py` (implementation todo #4).
2. Extend `Chimera/strategy.py` with barrier/turd helper methods referenced above.
3. Rework heuristic and pruning logic per sections 3–4.
4. Run match suites, record win/loss tallies, fine-tune weights.
