"""Trapdoor belief maintenance with mild temporal decay."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from game.game_map import prob_feel, prob_hear


@dataclass
class TrapdoorBelief:
    map_size: int = 8
    decay: float = 0.02
    priors: List[np.ndarray] = field(default_factory=list)
    probs: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._initialize_priors()
        self.probs = [p.copy() for p in self.priors]

    def _initialize_priors(self) -> None:
        weights = np.zeros((self.map_size, self.map_size))
        for x in range(self.map_size):
            for y in range(self.map_size):
                ring = min(min(x, self.map_size - 1 - x), min(y, self.map_size - 1 - y))
                if ring >= 3:
                    weight = 2.0
                elif ring == 2:
                    weight = 1.0
                elif ring == 1:
                    weight = 0.2
                else:
                    weight = 0.05
                weights[x, y] = weight

        self.priors = []
        for parity in (0, 1):
            mask = np.zeros_like(weights)
            mask[(np.indices(weights.shape).sum(axis=0) % 2) == parity] = 1.0
            prior = weights * mask
            total = prior.sum()
            self.priors.append(prior / total if total else mask / mask.sum())

    def decay_to_prior(self) -> None:
        for i in (0, 1):
            self.probs[i] = (
                (1.0 - self.decay) * self.probs[i]
                + self.decay * self.priors[i]
            )

    def update(self, loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]) -> None:
        self.decay_to_prior()
        for parity in (0, 1):
            heard, felt = sensor_data[parity]
            grid = self.probs[parity]
            updated = np.zeros_like(grid)
            total = 0.0
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if (x + y) % 2 != parity:
                        continue
                    prior = grid[x, y]
                    if prior == 0:
                        continue
                    dx = abs(x - loc[0])
                    dy = abs(y - loc[1])
                    p_hear = prob_hear(dx, dy)
                    p_feel = prob_feel(dx, dy)
                    likelihood = (p_hear if heard else 1 - p_hear) * (
                        p_feel if felt else 1 - p_feel
                    )
                    posterior = prior * likelihood
                    updated[x, y] = posterior
                    total += posterior
            if total == 0:
                # No new information; fall back to prior
                self.probs[parity] = self.priors[parity].copy()
            else:
                self.probs[parity] = updated / total

    def get_prob_at(self, loc: Tuple[int, int]) -> float:
        parity = (loc[0] + loc[1]) % 2
        return float(self.probs[parity][loc[0], loc[1]])
