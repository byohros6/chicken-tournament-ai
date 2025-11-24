from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from game.game_map import prob_feel, prob_hear

@dataclass
class TrapdoorBelief:
    map_size: int = 8
    decay: float = 0.05 # Forget 5% of fear every turn (Handle noise)
    priors: List[np.ndarray] = field(default_factory=list)
    probs: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._initialize_priors()
        self.probs = [p.copy() for p in self.priors]

    def _initialize_priors(self) -> None:
        # Center squares are more likely to be trapdoors
        weights = np.zeros((self.map_size, self.map_size))
        for x in range(self.map_size):
            for y in range(self.map_size):
                dist_x = min(x, self.map_size - 1 - x)
                dist_y = min(y, self.map_size - 1 - y)
                ring = min(dist_x, dist_y)
                
                if ring >= 3: weight = 2.0
                elif ring == 2: weight = 1.0
                elif ring == 1: weight = 0.2
                else: weight = 0.05
                weights[x, y] = weight

        self.priors = []
        for parity in (0, 1):
            p = weights.copy()
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if (x + y) % 2 != parity: p[x, y] = 0
            if np.sum(p) > 0: p /= np.sum(p)
            self.priors.append(p)

    def update(self, loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]) -> None:
        # Decay old beliefs slightly to prevent getting stuck
        for i in range(2):
            self.probs[i] = (1 - self.decay) * self.probs[i] + self.decay * self.priors[i]

        # Bayesian Update
        for parity in (0, 1):
            heard, felt = sensor_data[parity]
            grid = self.probs[parity]
            updated = np.zeros_like(grid)
            total = 0.0
            
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if (x + y) % 2 != parity: continue
                    prior = grid[x, y]
                    if prior == 0: continue
                    
                    dx = abs(x - loc[0])
                    dy = abs(y - loc[1])
                    p_h = prob_hear(dx, dy)
                    p_f = prob_feel(dx, dy)
                    
                    lh = p_h if heard else (1 - p_h)
                    lf = p_f if felt else (1 - p_f)
                    
                    post = prior * lh * lf
                    updated[x, y] = post
                    total += post
            
            if total > 0: self.probs[parity] = updated / total

    def get_prob_at(self, loc: Tuple[int, int]) -> float:
        return self.probs[(loc[0] + loc[1]) % 2][loc[0], loc[1]]