import numpy as np
from game.game_map import prob_hear, prob_feel

class Oracle:
    def __init__(self, map_size=8):
        self.map_size = map_size
        self.belief_white = np.zeros((map_size, map_size))
        self.belief_black = np.zeros((map_size, map_size))
        self._initialize_priors()

    def _initialize_priors(self):
        dim = self.map_size
        weights = np.zeros((dim, dim))
        weights[2 : dim - 2, 2 : dim - 2] = 1.0
        weights[3 : dim - 3, 3 : dim - 3] = 2.0
        
        total_weight = np.sum(weights)
        for r in range(dim):
            for c in range(dim):
                if (r + c) % 2 == 0:
                    self.belief_white[r, c] = weights[r, c]
                else:
                    self.belief_black[r, c] = weights[r, c]
        
        sum_white = np.sum(self.belief_white)
        sum_black = np.sum(self.belief_black)
        if sum_white > 0: self.belief_white /= sum_white
        if sum_black > 0: self.belief_black /= sum_black

    def update(self, current_loc, sensor_data):
        obs_white = sensor_data[0]
        obs_black = sensor_data[1]
        self._bayes_update(self.belief_white, current_loc, obs_white)
        self._bayes_update(self.belief_black, current_loc, obs_black)

    def _bayes_update(self, belief_grid, current_loc, observation):
        heard, felt = observation
        row, col = current_loc
        likelihood_mask = np.zeros_like(belief_grid)
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                if belief_grid[r, c] == 0: continue
                dx = abs(r - row)
                dy = abs(c - col)
                p_h = prob_hear(dx, dy)
                p_f = prob_feel(dx, dy)
                
                prob_obs = 1.0
                prob_obs *= p_h if heard else (1.0 - p_h)
                prob_obs *= p_f if felt else (1.0 - p_f)
                
                likelihood_mask[r, c] = prob_obs
        
        belief_grid *= likelihood_mask
        total = np.sum(belief_grid)
        if total > 0: belief_grid /= total

    def get_risk(self, loc):
        r, c = loc
        if (r + c) % 2 == 0: return self.belief_white[r, c]
        else: return self.belief_black[r, c]