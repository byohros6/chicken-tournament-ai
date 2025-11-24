"""Strategos – research-driven architecture.

This version implements the three-way split from the strategy report:

* Oracle      – maintains a trapdoor belief / risk map (simplified for now).
* GraphManager – exposes basic topology queries (reachability & mobility).
* Executive   – phase-based policy over the above (opening / mid / end).

Short‑term, this is still tuned first to *never* lose to Yolanda:

* Never enter the 4×4 danger zone (rings 2–3) – safe perimeter doctrine.
* Never place turds vs Yolanda (no self‑boxing risk, simpler state).
* Always egg when standing on a safe own‑parity square and an EGG move
  exists (egg‑on‑arrival rule).

Later, the same architecture can grow a full Bayesian Oracle, real graph
analysis, and smarter turd usage for strong opponents.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction


Coord = Tuple[int, int]
Move = Tuple[Direction, MoveType]


# ---------------------------------------------------------------------------
# Oracle – trapdoor risk / safe-perimeter model
# ---------------------------------------------------------------------------


class Oracle:
    """Simplified risk model consistent with the research doc.

    * Ring 0–1 (outer 48 squares) are absolutely safe.
    * Inner 4×4 danger zone (ring 2–3) is off‑limits in this version.

    We expose a binary API for the Executive: `is_safe(loc)`.
    """

    def __init__(self, size: int = 8):
        self.size = size

    def is_safe(self, loc: Coord) -> bool:
        r, c = loc
        # Ring index = min distance to any edge.
        ring = min(r, c, self.size - 1 - r, self.size - 1 - c)
        # Rings 0 and 1 are always safe per rules; rings >= 2 contain trapdoors.
        return ring <= 1


# ---------------------------------------------------------------------------
# GraphManager – light‑weight topology helper
# ---------------------------------------------------------------------------


@dataclass
class GraphManager:
    size: int

    def in_bounds(self, loc: Coord) -> bool:
        r, c = loc
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors4(self, loc: Coord) -> List[Coord]:
        r, c = loc
        cand = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [(rr, cc) for rr, cc in cand if self.in_bounds((rr, cc))]

    def mobility(self, board: Board, start: Coord) -> int:
        """Return count of reachable squares for *our* chicken.

        Engine already encodes egg/turd and adjacency rules for each player,
        so we just BFS over *moves we are allowed to make*.
        """

        seen = set()
        q: List[Coord] = [start]
        while q:
            cur = q.pop()
            if cur in seen:
                continue
            seen.add(cur)
            # Simulate stepping in each direction; rely on engine's
            # `get_valid_moves` from actual state in Executive instead when
            # used for evaluation. Here we just over‑approximate by neighbors
            # and let Executive do finer checks.
            for nxt in self.neighbors4(cur):
                if nxt not in seen:
                    q.append(nxt)
        return len(seen)


# ---------------------------------------------------------------------------
# Executive – phase‑based decision logic
# ---------------------------------------------------------------------------


class Executive:
    def __init__(self, size: int, my_parity: int, oracle: Oracle):
        self.size = size
        self.my_parity = my_parity
        self.oracle = oracle
        self.graph = GraphManager(size)
        # Short memory to discourage tiny cycles.
        self.history: Deque[Coord] = deque(maxlen=6)
        # Precompute all safe own-parity squares in the outer two rings.
        self.safe_own_parity: List[Coord] = []
        for r in range(size):
            for c in range(size):
                loc = (r, c)
                if not oracle.is_safe(loc):
                    continue
                if (r + c) % 2 != my_parity:
                    continue
                self.safe_own_parity.append(loc)

    # --- Public entry point -------------------------------------------------

    def choose_move(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Move:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return (Direction.UP, MoveType.PLAIN)

        my_loc = board.chicken_player.get_location()
        self.history.append(my_loc)

        turn = 80 - board.turns_left_player

        # Against Yolanda we run a single deterministic coverage policy.
        return self._coverage_policy(board, my_loc, valid_moves)

    # --- Single coverage policy --------------------------------------------

    def _coverage_policy(self, board: Board, my_loc: Coord, valid_moves: List[Move]) -> Move:
        """Deterministic safe-band coverage policy.

        1. Always egg on safe own-parity squares when possible.
        2. Otherwise, move greedily toward nearest un-egged safe own-parity.
        3. Fall back to any safe plain move if needed.
        """
        egg = self._egg_on_arrival(board, my_loc, valid_moves)
        if egg is not None:
            return egg
        move = self._greedy_safe_move_to_target(board, my_loc, valid_moves)
        if move is not None:
            return move
        return self._fallback_move(my_loc, valid_moves)

    # --- Core subroutines ---------------------------------------------------

    def _egg_on_arrival(
        self, board: Board, my_loc: Coord, valid_moves: List[Move]
    ) -> Optional[Move]:
        """If any EGG move takes us to a safe own‑parity square, take it.

        This encodes the "egg‑on‑arrival" doctrine and enforces perimeter safety.
        """

        best: Optional[Move] = None
        best_score = -1_000_000
        my_eggs = set(board.eggs_player)

        for direction, mtype in valid_moves:
            if mtype != MoveType.EGG:
                continue

            nxt = loc_after_direction(my_loc, direction)

            # Only consider eggs that land on our precomputed safe own-parity band.
            if nxt not in self.safe_own_parity:
                continue

            # Don't waste a move re‑egging the same square.
            if nxt in my_eggs:
                continue

            score = 0
            # Corners give a 3‑egg swing; treat them as very high value.
            if (nxt[0] in [0, self.size - 1]) and (nxt[1] in [0, self.size - 1]):
                score += 500
            elif nxt[0] in [0, self.size - 1] or nxt[1] in [0, self.size - 1]:
                score += 250
            else:
                score += 100

            # Mild preference for fresh territory.
            if nxt not in self.history:
                score += 20

            if score > best_score:
                best_score = score
                best = (direction, mtype)

        return best

    def _greedy_safe_move_to_target(
        self, board: Board, my_loc: Coord, valid_moves: List[Move]
    ) -> Optional[Move]:
        """Move greedily toward our best safe own‑parity target square.

        Target selection implements the "secure the perimeter" doctrine.
        """

        target = self._best_safe_target(board, my_loc)

        plain_moves = [m for m in valid_moves if m[1] == MoveType.PLAIN]
        if not plain_moves:
            return None

        best: Optional[Move] = None
        best_score = -1_000_000

        for direction, mtype in plain_moves:
            nxt = loc_after_direction(my_loc, direction)
            if not self.graph.in_bounds(nxt):
                continue
            if not self.oracle.is_safe(nxt):
                continue
            if board.is_cell_blocked(nxt):
                continue

            # Anti‑oscillation: avoid going back to the last square and avoid
            # very recent 2–3‑cycles when possible.
            if self.history and nxt == self.history[-1]:
                continue
            recent = list(self.history)[-3:]
            loop_penalty = 40 if nxt in recent else 0

            # Greedy step: prefer moves that strictly reduce distance to target.
            score = 0
            if target is not None:
                dist_now = abs(my_loc[0] - target[0]) + abs(my_loc[1] - target[1])
                dist_next = abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])

                if dist_next < dist_now:
                    score += 200
                elif dist_next == dist_now:
                    score += 10
                else:
                    score -= 50

            # Strong bias for staying on the outer ring.
            if nxt[0] in [0, self.size - 1] or nxt[1] in [0, self.size - 1]:
                score += 30

            score -= loop_penalty

            if score > best_score:
                best_score = score
                best = (direction, mtype)

        return best

    def _best_safe_target(self, board: Board, my_loc: Coord) -> Optional[Coord]:
        """Choose a high‑value safe own‑parity square to chase.

        Scoring follows the research report:
        * Corners >> edges >> interior, but all in the safe perimeter.
        * Only own‑parity squares without our egg already.
        """

        best: Optional[Coord] = None
        best_score = -1_000_000
        my_eggs = set(board.eggs_player)

        # Candidates are exactly the safe, own-parity band squares we
        # precomputed, minus any that already have our egg.
        for (r, c) in self.safe_own_parity:
            loc = (r, c)
            if loc in my_eggs:
                continue

            score = 0
            if (r in [0, self.size - 1]) and (c in [0, self.size - 1]):
                score += 500
            elif r in [0, self.size - 1] or c in [0, self.size - 1]:
                score += 250
            else:
                score += 100

            dist = abs(my_loc[0] - r) + abs(my_loc[1] - c)
            score -= dist * 6

            if score > best_score:
                best_score = score
                best = loc

        return best

    def _fallback_move(self, my_loc: Coord, valid_moves: List[Move]) -> Move:
        """Last‑resort move: any safe plain move, then any valid move."""

        plain_moves = [m for m in valid_moves if m[1] == MoveType.PLAIN]
        if plain_moves:
            for direction, mtype in plain_moves:
                nxt = loc_after_direction(my_loc, direction)
                if self.history and nxt == self.history[-1]:
                    continue
                return (direction, mtype)

        return valid_moves[0]


# ---------------------------------------------------------------------------
# Tournament entry point – PlayerAgent
# ---------------------------------------------------------------------------


class PlayerAgent:
    """Thin wrapper that wires Board/Oracle/Executive together.

    The tournament runner only knows about this class.
    """

    def __init__(self, board: Board, time_left: Callable[[], float]):
        size = board.game_map.MAP_SIZE
        my_loc = board.chicken_player.get_location()
        my_parity = (my_loc[0] + my_loc[1]) % 2

        # my_parity is 0 for even (our scoring lattice), 1 for odd.
        self.my_parity = my_parity
        self.oracle = Oracle(size)
        self.exec = Executive(size, my_parity, self.oracle)

    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Move:
        return self.exec.choose_move(board, sensor_data, time_left)

