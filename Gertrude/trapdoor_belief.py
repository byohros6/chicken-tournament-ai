import numpy as np
from game.game_map import prob_hear, prob_feel

class TrapdoorBelief:
    def __init__(self):
        self.MAP_SIZE = 8
        # We track two trapdoors independently:
        # index 0 = Even Parity Trapdoor (White)
        # index 1 = Odd Parity Trapdoor (Black)
        self.probs = [np.zeros((8, 8)), np.zeros((8, 8))]
        self._initialize_priors()

    def _initialize_priors(self):
        """
        Sets up the initial probability map based on the 'Ring' logic 
        described in the assignment. Center squares are more likely.
        """
        # Weights from assignment: Edge=0, Inner1=0, Inner2=1, Center=2
        weights = np.zeros((8, 8))
        for x in range(8):
            for y in range(8):
                dist_x = min(x, 7-x)
                dist_y = min(y, 7-y)
                ring = min(dist_x, dist_y)
                
                if ring == 3: weight = 2  # Center (3,3), (3,4), etc.
                elif ring == 2: weight = 1
                else: weight = 0.05 # Give edges a tiny non-zero chance just in case
                
                weights[x][y] = weight

        # Normalize for Even (0) and Odd (1) maps separately
        for parity in [0, 1]:
            total_weight = 0
            for x in range(8):
                for y in range(8):
                    if (x + y) % 2 == parity:
                        total_weight += weights[x][y]
                    else:
                        weights[x][y] = 0 # Impossible for this trapdoor to be here
            
            if total_weight > 0:
                self.probs[parity] = weights / total_weight

    def update(self, my_loc, sensor_data):
        """
        Update beliefs based on new sensor data using Bayes' Rule.
        sensor_data: [(hear_even, feel_even), (hear_odd, feel_odd)]
        """
        # Update both trapdoor beliefs independently
        for i in range(2):
            did_hear, did_feel = sensor_data[i]
            
            current_probs = self.probs[i]
            new_probs = np.zeros((8, 8))
            total_prob = 0

            for x in range(8):
                for y in range(8):
                    # Skip impossible squares (wrong color)
                    if (x + y) % 2 != i:
                        continue

                    prior = current_probs[x][y]
                    if prior == 0: continue

                    # Calculate Likelihood: P(Sensor | Trapdoor is at x,y)
                    dist_x = abs(x - my_loc[0])
                    dist_y = abs(y - my_loc[1])
                    
                    # Probability of hearing it given distance
                    p_h = prob_hear(dist_x, dist_y)
                    likelihood_hear = p_h if did_hear else (1 - p_h)
                    
                    # Probability of feeling it given distance
                    p_f = prob_feel(dist_x, dist_y)
                    likelihood_feel = p_f if did_feel else (1 - p_f)

                    # Bayes Update
                    posterior = prior * likelihood_hear * likelihood_feel
                    new_probs[x][y] = posterior
                    total_prob += posterior

            # Normalize so probabilities sum to 1
            if total_prob > 0:
                self.probs[i] = new_probs / total_prob

    def get_prob_at(self, loc):
        """Returns the probability that a trapdoor exists at loc (x,y)"""
        parity = (loc[0] + loc[1]) % 2
        return self.probs[parity][loc[0]][loc[1]]