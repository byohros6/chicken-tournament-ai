import numpy as np

# Exact logic from tournament engine
def get_sensor_probs(dx, dy):
    # Hearing (Engine logic)
    # "A square that shares an edge (1,0)... 50%"
    # "A square that is diagonal (1,1)... 25%"
    # "A square that shares an edge with (those)... 10%" -> Distance (2,0) or (0,2) or (2,1) etc?
    # Engine implementation:
    p_hear = 0.0
    if dx > 2 or dy > 2:
        p_hear = 0.0
    elif dx == 2 and dy == 2: # Corner of 5x5 grid
        p_hear = 0.0
    elif dx == 2 or dy == 2: # Edge of 5x5 grid
        p_hear = 0.1
    elif dx == 1 and dy == 1: # Diagonal
        p_hear = 0.25
    elif dx == 1 or dy == 1: # Adjacent (1,0) or (0,1)
        p_hear = 0.5
        
    # Feeling (Engine logic)
    p_feel = 0.0
    if dx > 1 or dy > 1:
        p_feel = 0.0
    elif dx == 1 and dy == 1: # Diagonal
        p_feel = 0.15
    elif dx == 1 or dy == 1: # Adjacent
        p_feel = 0.3
        
    return p_hear, p_feel

class TrapdoorBelief:
    def __init__(self):
        # Prior weights based on "Distance from Edge" rule
        weights = np.zeros((8,8))
        for r in range(8):
            for c in range(8):
                dist_edge = min(r, 7-r, c, 7-c)
                if dist_edge >= 3: w = 2.0
                elif dist_edge == 2: w = 1.0
                else: w = 0.0
                weights[r,c] = w
        
        s = np.sum(weights)
        # Avoid divide by zero if weights sum to 0 (impossible here but safe)
        self.probs = weights / s if s > 0 else np.ones((8,8))/64

    def update(self, my_loc, sensor_data):
        # sensor_data: [(hear_white, feel_white), (hear_black, feel_black)]
        for parity in [0, 1]:
            heard, felt = sensor_data[parity]
            
            for r in range(8):
                for c in range(8):
                    # Only update squares matching the trap's color
                    if (r+c)%2 != parity:
                        self.probs[r,c] = 0.0
                        continue
                        
                    dx, dy = abs(r-my_loc[0]), abs(c-my_loc[1])
                    p_h, p_f = get_sensor_probs(dx, dy)
                    
                    # Likelihood
                    l_hear = p_h if heard else (1 - p_h)
                    l_feel = p_f if felt else (1 - p_f)
                    
                    self.probs[r,c] *= (l_hear * l_feel)
            
            # Normalize the parity group to keep probabilities valid
            mask = (np.indices((8,8)).sum(axis=0) % 2) == parity
            total = np.sum(self.probs[mask])
            if total > 0:
                self.probs[mask] /= total