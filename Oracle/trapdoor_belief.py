import numpy as np

class TrapdoorBelief:
    def __init__(self, board_width, board_height):
        self.width = board_width
        self.height = board_height
        
        self.even_belief = self._initialize_prior(is_even=True)
        self.odd_belief = self._initialize_prior(is_even=False)

    def _initialize_prior(self, is_even):
        grid = np.zeros((self.width, self.height))
        
        for x in range(self.width):
            for y in range(self.height):
                if ((x + y) % 2 == 0) != is_even:
                    continue

                dist_x = min(x, self.width - 1 - x)
                dist_y = min(y, self.height - 1 - y)
                dist_edge = min(dist_x, dist_y)
                
                # Map distance to weight based on rules
                # 0 (edge) -> weight 0
                # 1 (next ring) -> weight 0
                # 2 (next ring) -> weight 1
                # 3 (center) -> weight 2
                if dist_edge == 0:
                    weight = 0.01 
                elif dist_edge == 1:
                    weight = 0.01
                elif dist_edge == 2:
                    weight = 1.0
                else:
                    weight = 2.0
                
                grid[x, y] = weight
        
        # Normalize so sum is 1.0
        total = np.sum(grid)
        if total > 0:
            grid /= total
        
        return grid

    def update(self, current_pos, sensor_data):
        heard_even, felt_even = sensor_data[0]
        heard_odd, felt_odd = sensor_data[1]

        self.even_belief = self._bayes_update(
            self.even_belief, current_pos, heard_even, felt_even, is_even=True
        )

        self.odd_belief = self._bayes_update(
            self.odd_belief, current_pos, heard_odd, felt_odd, is_even=False
        )

    def _bayes_update(self, prior, current_pos, heard, felt, is_even):
        likelihood_grid = np.zeros_like(prior)
        
        cx, cy = current_pos
        
        for tx in range(self.width):
            for ty in range(self.height):
                if prior[tx, ty] == 0:
                    continue
                    
                prob_h, prob_f = self._get_sensor_probs((cx, cy), (tx, ty))
                
                p_obs_h = prob_h if heard else (1 - prob_h)
                p_obs_f = prob_f if felt else (1 - prob_f)
                
                likelihood = p_obs_h * p_obs_f
                likelihood_grid[tx, ty] = likelihood

        posterior = prior * likelihood_grid
        
        total = np.sum(posterior)
        if total == 0:
            return prior 
        
        return posterior / total

    def _get_sensor_probs(self, curr, trap):
        dx = abs(curr[0] - trap[0])
        dy = abs(curr[1] - trap[1])
        
        if dx + dy == 1:
            return 0.50, 0.30
        if dx == 1 and dy == 1:
            return 0.25, 0.15
        
        if (dx == 0 and dy == 2) or (dx == 2 and dy == 0) or \
           (dx == 1 and dy == 2) or (dx == 2 and dy == 1):
            return 0.10, 0.00
        
        return 0.00, 0.00

    def get_risk(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
            
        if (x + y) % 2 == 0:
            return self.even_belief[x, y]
        else:
            return self.odd_belief[x, y]
            
    def normalize_after_safe_move(self, x, y):
        if (x + y) % 2 == 0:
            self.even_belief[x, y] = 0
            total = np.sum(self.even_belief)
            if total > 0: self.even_belief /= total
        else:
            self.odd_belief[x, y] = 0
            total = np.sum(self.odd_belief)
            if total > 0: self.odd_belief /= total