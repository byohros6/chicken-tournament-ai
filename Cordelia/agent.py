from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction

from .belief import TrapdoorBelief

Move = Tuple[Direction, MoveType]


class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable[[], float]):
        # Calm Gertrude unless explicitly re-enabled
        os.environ.setdefault("MUTE_GERTRUDE", "1")

        self.trap_belief = TrapdoorBelief()
        self.last_best: Optional[Move] = None
        self.killer_moves: Dict[int, Move] = {}
        self.time_buffer = 0.02
        self.recent_locs: Deque[Tuple[int, int]] = deque(maxlen=6)
        self.chatty = True
        self.barrier_targets = self._generate_barrier_targets(board)
        self.barrier_cursor = 0
        self.current_phase = "barrier"
        self.known_traps: Set[Tuple[int, int]] = set()
        self.failed_targets: Set[Tuple[int, int]] = set()
        self.barrier_turn_cap = 24
        self.trap_block_threshold = 0.18
        self.trap_step_threshold = 0.3

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Move:
        start = time.time()
        self.trap_belief.update(board.chicken_player.get_location(), sensor_data)
        self._ingest_new_traps(board)
        self._sync_barrier_progress(board)
        self.current_phase = self._phase(board)
        time_budget = self._choose_time_budget(time_left())
        forced = self._forced_plan_move(board)
        if forced is not None:
            self._log("Executing forced plan move to advance barrier.")
            return forced

        best_move = self._fallback_move(board)
        best_score = float("-inf")
        depth = 1

        self._track_position(board.chicken_player.get_location())

        while True:
            if self._should_abort(start, time_budget):
                break
            score, move = self._search(board, depth, -math.inf, math.inf, True, start, time_budget)
            if move is None:
                break
            best_move, best_score = move, score
            self.last_best = move
            depth += 1
            # Cap depth to avoid runaway on quiet boards
            if depth > 12:
                break

        best_move = self._resolve_loop(board, best_move)
        reason = self._build_reason(board, best_move, best_score)

        elapsed = time.time() - start
        self._log(
            f"depth {depth-1} | score {best_score:.1f} | move {best_move} | "
            f"{elapsed:.3f}s used / {time_budget:.3f}s budget. {reason}"
        )
        return best_move

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _search(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        start: float,
        budget: float,
    ) -> Tuple[float, Optional[Move]]:
        if self._should_abort(start, budget):
            return self._evaluate(board), None
        if depth == 0 or board.is_game_over():
            return self._evaluate(board), None

        moves = board.get_valid_moves(enemy=not maximizing)
        if maximizing:
            moves = self._filter_trap_averse_moves(board, moves)
        if not moves:
            return self._evaluate(board), None

        ordered_moves = self._order_moves(moves, depth, maximizing)
        best_move: Optional[Move] = None

        if maximizing:
            value = -math.inf
            for move in ordered_moves:
                next_board = board.forecast_move(*move, check_ok=False)
                if next_board is None:
                    continue
                score, _ = self._search(next_board, depth - 1, alpha, beta, False, start, budget)
                if self._should_abort(start, budget):
                    return value if best_move else self._evaluate(board), best_move
                if score > value:
                    value, best_move = score, move
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.killer_moves[depth] = move
                    break
            return value, best_move
        else:
            value = math.inf
            for move in ordered_moves:
                next_board = board.forecast_move(*move, check_ok=False)
                if next_board is None:
                    continue
                score, _ = self._search(next_board, depth - 1, alpha, beta, True, start, budget)
                if self._should_abort(start, budget):
                    return value if best_move else self._evaluate(board), best_move
                if score < value:
                    value, best_move = score, move
                beta = min(beta, value)
                if alpha >= beta:
                    self.killer_moves[depth] = move
                    break
            return value, best_move

    def _order_moves(self, moves: List[Move], depth: int, maximizing: bool) -> List[Move]:
        def score(move: Move) -> Tuple[int, int]:
            direction, move_type = move
            killer_bonus = 1 if self.killer_moves.get(depth) == move else 0
            type_score = {
                MoveType.EGG: 0,
                MoveType.PLAIN: 1,
                MoveType.TURD: 2,
            }[move_type]
            phase_bias = self._phase_move_bias(move_type)
            # maximizing wants low tuple
            return (type_score - phase_bias, -killer_bonus)

        ordered = sorted(moves, key=score)
        # Try last successful move first if available
        if self.last_best and self.last_best in ordered:
            ordered.remove(self.last_best)
            ordered.insert(0, self.last_best)
        return ordered

    def _fallback_move(self, board: Board) -> Move:
        moves = board.get_valid_moves()
        moves = self._filter_trap_averse_moves(board, moves)
        if not moves:
            return Direction.UP, MoveType.PLAIN

        egg_deficit = board.chicken_enemy.get_eggs_laid() - board.chicken_player.get_eggs_laid()
        urgent = egg_deficit >= 0 or board.turns_left_player < 10
        egg_moves = [mv for mv in moves if mv[1] == MoveType.EGG]
        if urgent and egg_moves:
            return egg_moves[0]

        scoring: List[Tuple[float, Move]] = []
        for mv in moves:
            next_board = board.forecast_move(*mv, check_ok=False)
            if next_board is None:
                continue
            scoring.append((self._evaluate(next_board), mv))
        if scoring:
            scoring.sort(key=lambda pair: pair[0], reverse=True)
            return scoring[0][1]
        return moves[0]

    def _should_abort(self, start: float, budget: float) -> bool:
        return (time.time() - start) >= max(0.01, budget - self.time_buffer)

    def _choose_time_budget(self, time_remaining: float) -> float:
        # Spend up to 0.9s early, but tighten when low on clock
        urgency = 0.15 if time_remaining > 90 else 0.08 if time_remaining > 30 else 0.05
        return min(1.0, max(0.15, time_remaining * urgency))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, board: Board) -> float:
        me = board.chicken_player
        enemy = board.chicken_enemy
        my_loc = me.get_location()
        enemy_loc = enemy.get_location()

        egg_diff = me.get_eggs_laid() - enemy.get_eggs_laid()
        deficit = -egg_diff if egg_diff < 0 else 0
        urgency = 1.0 + deficit * 0.35 + max(0, 12 - board.turns_left_player) * 0.04

        score = egg_diff * 1100 * urgency

        my_moves = len(board.get_valid_moves(enemy=False))
        enemy_moves = len(board.get_valid_moves(enemy=True))
        score += (my_moves - enemy_moves) * 55

        phase = self._phase(board)
        score += self._egg_setup_bonus(board, deficit)
        score -= self._enemy_egg_pressure(board)
        score += self._phase_bias(board, phase)

        trap_penalty = self._trap_penalty(board, my_loc, deficit)
        score -= trap_penalty

        score += self._territory_pressure(board, my_loc, enemy_loc, deficit)

        if board.is_cell_in_enemy_turd_zone(my_loc):
            score -= 180

        score += self._corner_pull(me, my_loc) * 0.6
        score -= self._corner_pull(enemy, enemy_loc) * 0.3

        score -= self._distance_to_center(my_loc) * 25
        score += self._distance_to_center(enemy_loc) * 18

        manhattan = abs(my_loc[0] - enemy_loc[0]) + abs(my_loc[1] - enemy_loc[1])
        score -= manhattan * (12 + (10 if deficit > 0 else 0))

        score += (5 - me.get_turds_left()) * 10
        score -= (5 - enemy.get_turds_left()) * 22

        return score

    def _corner_pull(self, chicken, loc: Tuple[int, int]) -> float:
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        valid = [c for c in corners if (c[0] + c[1]) % 2 == chicken.even_chicken]
        if not valid:
            return 0.0
        best = min(abs(loc[0] - cx) + abs(loc[1] - cy) for cx, cy in valid)
        value = max(0, 6 - best) * 35
        if loc in valid:
            value += 420
        return value

    def _distance_to_center(self, loc: Tuple[int, int]) -> float:
        center = 3.5
        return abs(loc[0] - center) + abs(loc[1] - center)

    def _egg_site_distance(self, board: Board, for_enemy: bool = False) -> Optional[int]:
        chicken = board.chicken_enemy if for_enemy else board.chicken_player
        eggs = board.eggs_enemy if for_enemy else board.eggs_player
        my_turds = board.turds_enemy if for_enemy else board.turds_player
        opp_turds = board.turds_player if for_enemy else board.turds_enemy
        origin = chicken.get_location()
        size = board.game_map.MAP_SIZE
        best: Optional[int] = None
        for x in range(size):
            for y in range(size):
                loc = (x, y)
                if not self._can_lay_on_loc(loc, chicken.even_chicken, eggs, my_turds, opp_turds):
                    continue
                dist = abs(loc[0] - origin[0]) + abs(loc[1] - origin[1])
                if best is None or dist < best:
                    best = dist
        return best

    def _can_lay_on_loc(
        self,
        loc: Tuple[int, int],
        parity: int,
        eggs: set,
        own_turds: set,
        enemy_turds: set,
    ) -> bool:
        if (loc[0] + loc[1]) % 2 != parity:
            return False
        if loc in eggs or loc in own_turds or loc in enemy_turds:
            return False
        return True

    # ------------------------------------------------------------------
    # Strategy planning helpers
    # ------------------------------------------------------------------
    def _ingest_new_traps(self, board: Board) -> None:
        found = set(getattr(board, "found_trapdoors", set()))
        newly = found - self.known_traps
        if newly:
            self.failed_targets.update(newly)
            self.known_traps.update(newly)

    def _is_bad_barrier_cell(self, loc: Tuple[int, int], board: Board) -> bool:
        if loc in self.failed_targets:
            return True
        if self._is_known_trap(loc, board):
            return True
        if board.turn_count < self.barrier_turn_cap:
            if self._is_risky_trap_cell(loc, board, strict=True):
                return True
        return False

    def _plan_step_score(self, loc: Tuple[int, int], target: Tuple[int, int]) -> float:
        dist = abs(loc[0] - target[0]) + abs(loc[1] - target[1])
        trap_risk = self.trap_belief.get_prob_at(loc) * 10.0
        return dist + trap_risk

    def _best_plan_step(
        self, board: Board, target: Tuple[int, int], moves: List[Move]
    ) -> Optional[Move]:
        best_move: Optional[Move] = None
        best_score: Optional[float] = None
        for mv in moves:
            if mv[1] != MoveType.PLAIN:
                continue
            nxt = board.forecast_move(*mv, check_ok=False)
            if nxt is None:
                continue
            loc = nxt.chicken_player.get_location()
            if self._is_bad_barrier_cell(loc, board):
                continue
            score = self._plan_step_score(loc, target)
            if best_score is None or score < best_score:
                best_score = score
                best_move = mv
        return best_move

    def _choose_safe_turd_move(self, board: Board, moves: List[Move]) -> Optional[Move]:
        choice: Optional[Move] = None
        lowest_risk: Optional[float] = None
        for mv in moves:
            nxt = board.forecast_move(*mv, check_ok=False)
            if nxt is None:
                continue
            loc = nxt.chicken_player.get_location()
            if self._is_bad_barrier_cell(loc, board):
                continue
            risk = self.trap_belief.get_prob_at(loc)
            if lowest_risk is None or risk < lowest_risk:
                lowest_risk = risk
                choice = mv
        return choice

    def _project_location(self, board: Board, move: Move, enemy: bool = False) -> Tuple[int, int]:
        direction, _ = move
        chicken = board.chicken_enemy if enemy else board.chicken_player
        return loc_after_direction(chicken.get_location(), direction)

    def _is_known_trap(self, loc: Tuple[int, int], board: Board) -> bool:
        return loc in self.known_traps or loc in getattr(board, "found_trapdoors", set())

    def _trap_threshold(self, board: Board, strict: bool) -> float:
        base = self.trap_block_threshold if strict else self.trap_step_threshold
        deficit = board.chicken_enemy.get_eggs_laid() - board.chicken_player.get_eggs_laid()
        if deficit >= 1:
            base += min(0.12, 0.05 * deficit)
        if board.turns_left_player <= 8:
            base += 0.08
        return min(0.9, base)

    def _is_risky_trap_cell(self, loc: Tuple[int, int], board: Board, strict: bool = False) -> bool:
        if self._is_known_trap(loc, board):
            return True
        threshold = self._trap_threshold(board, strict)
        return self.trap_belief.get_prob_at(loc) >= threshold

    def _filter_trap_averse_moves(self, board: Board, moves: List[Move]) -> List[Move]:
        if not moves:
            return moves
        safe: List[Move] = []
        risky: List[Move] = []
        doomed: List[Move] = []
        for mv in moves:
            destination = self._project_location(board, mv)
            if self._is_known_trap(destination, board):
                doomed.append(mv)
                continue
            if self._is_risky_trap_cell(destination, board):
                risky.append(mv)
            else:
                safe.append(mv)
        if safe:
            return safe
        if risky:
            return risky
        return doomed if doomed else moves

    def _generate_barrier_targets(self, board: Board) -> List[Tuple[int, int]]:
        my_spawn = board.chicken_player.get_spawn()
        enemy_spawn = board.chicken_enemy.get_spawn()
        if not my_spawn or not enemy_spawn:
            return []
        dx = 1 if enemy_spawn[0] > my_spawn[0] else -1
        dy = 1 if enemy_spawn[1] > my_spawn[1] else -1
        cur = my_spawn
        targets = []
        budget = min(getattr(board.game_map, "MAX_TURDS", 5), 5)
        for _ in range(budget):
            cur = (cur[0] + dx, cur[1] + dy)
            if 0 <= cur[0] < board.game_map.MAP_SIZE and 0 <= cur[1] < board.game_map.MAP_SIZE:
                targets.append(cur)
        return targets

    def _sync_barrier_progress(self, board: Board) -> None:
        if not self.barrier_targets:
            return
        while self.barrier_cursor < len(self.barrier_targets):
            target = self.barrier_targets[self.barrier_cursor]
            if target in board.turds_player:
                self.barrier_cursor += 1
            elif self._is_bad_barrier_cell(target, board):
                self.failed_targets.add(target)
                self.barrier_cursor += 1
            else:
                break

    def _next_barrier_target(self, board: Board) -> Optional[Tuple[int, int]]:
        if not self.barrier_targets:
            return None
        idx = self.barrier_cursor
        while idx < len(self.barrier_targets):
            target = self.barrier_targets[idx]
            if target in board.turds_player:
                idx += 1
                continue
            if self._is_bad_barrier_cell(target, board):
                self.failed_targets.add(target)
                idx += 1
                continue
            self.barrier_cursor = idx
            return target
        self.barrier_cursor = len(self.barrier_targets)
        return None

    def _barrier_completion(self, board: Board) -> float:
        if not self.barrier_targets:
            return 1.0
        built = sum(1 for cell in self.barrier_targets if cell in board.turds_player)
        return built / len(self.barrier_targets)

    def _phase(self, board: Board) -> str:
        completion = self._barrier_completion(board)
        target_available = self._next_barrier_target(board)
        if (
            completion < 0.8
            and self.barrier_targets
            and target_available is not None
            and board.turn_count < self.barrier_turn_cap
        ):
            return "barrier"
        if board.turns_left_player <= 10:
            return "harvest"
        return "invasion"

    def _forced_plan_move(self, board: Board) -> Optional[Move]:
        if self.current_phase != "barrier":
            return None
        if board.turn_count >= self.barrier_turn_cap:
            return None
        target = self._next_barrier_target(board)
        if target is None:
            return None
        moves = board.get_valid_moves()
        if board.chicken_player.get_location() == target:
            turd_moves = [mv for mv in moves if mv[1] == MoveType.TURD]
            if turd_moves:
                preferred = self._choose_safe_turd_move(board, turd_moves)
                return preferred or turd_moves[0]
            return None
        step = self._best_plan_step(board, target, moves)
        if step:
            return step
        return None

    def _phase_move_bias(self, move_type: MoveType) -> int:
        if self.current_phase == "barrier":
            if move_type == MoveType.TURD:
                return 2
            if move_type == MoveType.EGG:
                return -1
        if self.current_phase == "harvest":
            if move_type == MoveType.EGG:
                return 1
            if move_type == MoveType.TURD:
                return -1
        if self.current_phase == "invasion" and move_type == MoveType.EGG:
            return 1
        return 0

    def _phase_bias(self, board: Board, phase: str) -> float:
        if phase == "barrier":
            return self._barrier_bias(board)
        if phase == "harvest":
            return self._harvest_bias(board)
        return self._invasion_bias(board)

    def _barrier_bias(self, board: Board) -> float:
        completion = self._barrier_completion(board)
        bonus = completion * 900
        target = self._next_barrier_target(board)
        if target:
            my_loc = board.chicken_player.get_location()
            dist = abs(my_loc[0] - target[0]) + abs(my_loc[1] - target[1])
            bonus += max(0, 10 - dist) * 70
        return bonus

    def _invasion_bias(self, board: Board) -> float:
        my_spawn = board.chicken_player.get_spawn()
        enemy_spawn = board.chicken_enemy.get_spawn()
        if not my_spawn or not enemy_spawn:
            return 0.0
        axis = 1 if enemy_spawn[0] > my_spawn[0] else -1
        my_loc = board.chicken_player.get_location()
        infiltration = max(0, (my_loc[0] - my_spawn[0]) * axis)
        enemy_half_eggs = sum(1 for egg in board.eggs_player if (egg[0] - my_spawn[0]) * axis > 0)
        return infiltration * 90 + enemy_half_eggs * 220

    def _harvest_bias(self, board: Board) -> float:
        my_spawn = board.chicken_player.get_spawn()
        enemy_spawn = board.chicken_enemy.get_spawn()
        if not my_spawn or not enemy_spawn:
            return 0.0
        axis = 1 if enemy_spawn[0] > my_spawn[0] else -1
        my_loc = board.chicken_player.get_location()
        away = max(0, (my_loc[0] - my_spawn[0]) * axis)
        open_sites = self._count_home_sites(board)
        return open_sites * 45 - away * 120

    def _count_home_sites(self, board: Board) -> int:
        my_spawn = board.chicken_player.get_spawn()
        enemy_spawn = board.chicken_enemy.get_spawn()
        if not my_spawn or not enemy_spawn:
            return 0
        axis = 1 if enemy_spawn[0] > my_spawn[0] else -1
        parity = board.chicken_player.even_chicken
        count = 0
        for x in range(board.game_map.MAP_SIZE):
            for y in range(board.game_map.MAP_SIZE):
                loc = (x, y)
                if ((x - my_spawn[0]) * axis) > 0:
                    continue
                if (x + y) % 2 != parity:
                    continue
                if loc in board.eggs_player or loc in board.turds_player:
                    continue
                count += 1
        return count

    # ------------------------------------------------------------------
    # Loop avoidance & narration helpers
    # ------------------------------------------------------------------
    def _track_position(self, loc: Tuple[int, int]) -> None:
        if not self.recent_locs or self.recent_locs[-1] != loc:
            self.recent_locs.append(loc)

    def _resolve_loop(self, board: Board, move: Move) -> Move:
        if move is None or len(self.recent_locs) < 2:
            return move
        prev = self.recent_locs[-2]
        current = self.recent_locs[-1]
        forecast = board.forecast_move(*move, check_ok=False)
        if forecast is None:
            return move
        next_loc = forecast.chicken_player.get_location()
        if next_loc == prev and current == self.recent_locs[-1]:
            alternative = self._choose_alternative(board, exclude=move)
            if alternative != move:
                self._log("Detected oscillation; diverting to alternate plan.")
                return alternative
        return move

    def _choose_alternative(self, board: Board, exclude: Move) -> Move:
        candidates = [mv for mv in board.get_valid_moves() if mv != exclude]
        candidates = self._filter_trap_averse_moves(board, candidates)
        best_move = exclude
        best_score = float("-inf")
        for mv in candidates:
            nxt = board.forecast_move(*mv, check_ok=False)
            if nxt is None:
                continue
            val = self._evaluate(nxt)
            if val > best_score:
                best_score = val
                best_move = mv
        return best_move

    def _build_reason(self, board: Board, move: Move, score: float) -> str:
        me = board.chicken_player
        enemy = board.chicken_enemy
        egg_diff = me.get_eggs_laid() - enemy.get_eggs_laid()
        trap_risk = self.trap_belief.get_prob_at(me.get_location())
        pieces = [f"eggs {egg_diff:+}", f"risk {trap_risk:.2f}"]
        if move and move[1] == MoveType.EGG:
            pieces.append("laying egg")
        elif move and move[1] == MoveType.TURD:
            pieces.append("dropping cover")
        else:
            pieces.append("maneuvering")
        pieces.append(f"eval {score:.0f}")
        return " | ".join(pieces)

    def _log(self, message: str) -> None:
        if self.chatty:
            print(f"[Cordelia] {message}")

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------
    def _egg_setup_bonus(self, board: Board, deficit: int) -> float:
        bonus = 0.0
        if board.can_lay_egg():
            bonus += 520 + deficit * 200
        dist_to_site = self._egg_site_distance(board)
        if dist_to_site is not None:
            bonus += max(0, 8 - dist_to_site) * (90 + deficit * 12)
        return bonus

    def _enemy_egg_pressure(self, board: Board) -> float:
        dist = self._egg_site_distance(board, for_enemy=True)
        if dist is None:
            return 0.0
        penalty = max(0, 8 - dist) * 95
        if dist == 0:
            penalty += 420
        return penalty

    def _trap_penalty(self, board: Board, loc: Tuple[int, int], deficit: int) -> float:
        risk = self.trap_belief.get_prob_at(loc)
        stage = 1.0 - (board.turns_left_player / board.MAX_TURNS)
        bravery = min(0.85, deficit * 0.25 + stage * 0.35)
        caution = max(0.2, 1.0 - bravery)
        base = 220 + 520 * caution
        return risk * base

    def _territory_pressure(
        self,
        board: Board,
        my_loc: Tuple[int, int],
        enemy_loc: Tuple[int, int],
        deficit: int,
    ) -> float:
        enemy_spawn = board.chicken_enemy.get_spawn()
        my_spawn = board.chicken_player.get_spawn()
        toward_enemy = max(
            0,
            12 - (abs(my_loc[0] - enemy_spawn[0]) + abs(my_loc[1] - enemy_spawn[1])),
        )
        toward_enemy *= 32 + deficit * 8
        enemy_pressure = max(
            0,
            12 - (abs(enemy_loc[0] - my_spawn[0]) + abs(enemy_loc[1] - my_spawn[1])),
        )
        enemy_pressure *= 24
        return toward_enemy - enemy_pressure
