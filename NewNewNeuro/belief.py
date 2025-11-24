import numpy as np

def get_sensor_probs(dx, dy):
    # Exact logic from tournament engine
    p_hear = 0.0
    if dx > 2 or dy > 2: p_hear = 0.0
    elif dx == 2 and dy == 2: p_hear = 0.0
    elif dx == 2 or dy == 2: p_hear = 0.1
    elif dx == 1 and dy == 1: p_hear = 0.25
    elif dx == 1 or dy == 1: p_hear = 0.5
        
    p_feel = 0.0
    if dx > 1 or dy > 1: p_feel = 0.0
    elif dx == 1 and dy == 1: p_feel = 0.15
    elif dx == 1 or dy == 1: p_feel = 0.3
        
    return p_hear, p_feel

class TrapdoorBelief:
    def __init__(self):
        # Prior weights (Center Heavy)
        weights = np.zeros((8,8))
        for r in range(8):
            for c in range(8):
                dist_edge = min(r, 7-r, c, 7-c)
                if dist_edge >= 3: w = 2.0
                elif dist_edge == 2: w = 1.0
                else: w = 0.0
                weights[r,c] = w
        
        s = np.sum(weights)
        self.probs = weights / s if s > 0 else np.ones((8,8))/64

    def update(self, my_loc, sensor_data):
        # sensor_data: [(hear_w, feel_w), (hear_b, feel_b)]
        for parity in [0, 1]:
            heard, felt = sensor_data[parity]
            
            for r in range(8):
                for c in range(8):
                    if (r+c)%2 != parity:
                        self.probs[r,c] = 0.0
                        continue
                        
                    dx, dy = abs(r-my_loc[0]), abs(c-my_loc[1])
                    p_h, p_f = get_sensor_probs(dx, dy)
                    
                    # Likelihood
                    l_hear = p_h if heard else (1 - p_h)
                    l_feel = p_f if felt else (1 - p_f)
                    
                    self.probs[r,c] *= (l_hear * l_feel)
            
            # Normalize
            mask = (np.indices((8,8)).sum(axis=0) % 2) == parity
            total = np.sum(self.probs[mask])
            if total > 0:
                self.probs[mask] /= total

    def set_confirmed_traps(self, locations):
        """
        Forces probability to 1.0 for known traps and 0.0 for others of same parity.
        This prevents the agent from walking into the same trap twice.
        """
        for (r, c) in locations:
            parity = (r + c) % 2
            # Set ALL squares of this parity to 0 (Since there is only 1 trap per color)
            mask = (np.indices((8,8)).sum(axis=0) % 2) == parity
            self.probs[mask] = 0.0
            # Set THE TRAP to 1.0 (Danger!)
            self.probs[r, c] = 1.0