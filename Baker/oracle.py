import numpy as np
from game.game_map import prob_hear, prob_feel

class Oracle:
    """
    A Bayesian Filter that estimates the probability of trapdoors being located
    at specific squares on the board.
    
    It maintains two belief states:
    - belief_white: Probability distribution for the trapdoor on white squares.
    - belief_black: Probability distribution for the trapdoor on black squares.
    """

    def __init__(self, map_size=8):
        self.map_size = map_size
        self.belief_white = np.zeros((map_size, map_size))
        self.belief_black = np.zeros((map_size, map_size))
        
        # Initialize priors based on the ring weights defined in trapdoor_manager.py
        self._initialize_priors()

    def _initialize_priors(self):
        """
        Sets the initial belief state based on the generation weights:
        - Outer Ring (0): Weight 0
        - Ring 1: Weight 0
        - Ring 2: Weight 1
        - Center (3x3 area effectively): Weight 2
        
        Reference: trapdoor_manager.py choose_trapdoors
        """
        dim = self.map_size
        weights = np.zeros((dim, dim))
        
        # Ring 2: 2 squares in from edge. Index range [2, dim-2)
        # We fill the inner box with 1.0 first
        weights[2 : dim - 2, 2 : dim - 2] = 1.0
        
        # Ring 3 (Center): 3 squares in. Index range [3, dim-3)
        # We overwrite the center with 2.0
        weights[3 : dim - 3, 3 : dim - 3] = 2.0
        
        # Normalize to create a valid probability distribution
        total_weight = np.sum(weights)
        
        # Split into white/black masks
        # White squares: (row + col) is even
        # Black squares: (row + col) is odd
        for r in range(dim):
            for c in range(dim):
                if (r + c) % 2 == 0:
                    self.belief_white[r, c] = weights[r, c]
                else:
                    self.belief_black[r, c] = weights[r, c]
        
        # Re-normalize each distribution independently
        # We divide by the sum of weights on THAT color to ensure P(sum) = 1.0
        sum_white = np.sum(self.belief_white)
        sum_black = np.sum(self.belief_black)
        
        if sum_white > 0:
            self.belief_white /= sum_white
        if sum_black > 0:
            self.belief_black /= sum_black

    def update(self, current_loc, sensor_data):
        """
        Updates the belief state based on new observations.
        
        Args:
            current_loc: Tuple(int, int) of the agent's current position.
            sensor_data: List of two tuples [(hear_white, feel_white), (hear_black, feel_black)]
                         Note: The rules/board send data as [trapdoor_white_sensor, trapdoor_black_sensor]
        """
        # Unpack sensor data
        # sensor_data[0] corresponds to the White trapdoor (even parity)
        obs_white = sensor_data[0] # (heard, felt)
        
        # sensor_data[1] corresponds to the Black trapdoor (odd parity)
        obs_black = sensor_data[1] # (heard, felt)
        
        # Update White Belief
        self._bayes_update(self.belief_white, current_loc, obs_white)
        
        # Update Black Belief
        self._bayes_update(self.belief_black, current_loc, obs_black)

    def _bayes_update(self, belief_grid, current_loc, observation):
        """
        Performs the math: P(H|E) = P(E|H) * P(H) / P(E)
        """
        heard, felt = observation
        row, col = current_loc
        
        # We iterate over every possible square 's' where the trapdoor could be.
        # For each 's', we calculate the probability of obtaining the observation 
        # (heard, felt) GIVEN the trapdoor is at 's'.
        
        likelihood_mask = np.zeros_like(belief_grid)
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                # If probability is already 0 (impossible), skip to save time
                if belief_grid[r, c] == 0:
                    continue
                
                # Calculate distance components
                dx = abs(r - row)
                dy = abs(c - col)
                
                # Get P(Heard | Distance) and P(Felt | Distance)
                p_h = prob_hear(dx, dy)
                p_f = prob_feel(dx, dy)
                
                # Probability of the specific observation
                # Since hear and feel are independent given location:
                prob_obs_given_loc = 1.0
                
                if heard:
                    prob_obs_given_loc *= p_h
                else:
                    prob_obs_given_loc *= (1.0 - p_h)
                    
                if felt:
                    prob_obs_given_loc *= p_f
                else:
                    prob_obs_given_loc *= (1.0 - p_f)
                
                likelihood_mask[r, c] = prob_obs_given_loc
        
        # Apply the update: Posterior = Prior * Likelihood
        belief_grid *= likelihood_mask
        
        # Normalize
        total = np.sum(belief_grid)
        if total > 0:
            belief_grid /= total
        else:
            # This should theoretically never happen unless sensors are broken
            # or the trapdoor is in an impossible location (Outer Ring).
            # Reset to uniform over non-zero priors if we crash here.
            pass

    def get_risk(self, loc):
        """
        Returns the probability (0.0 to 1.0) that a trapdoor is at `loc`.
        """
        r, c = loc
        if (r + c) % 2 == 0:
            return self.belief_white[r, c]
        else:
            return self.belief_black[r, c]

    def get_highest_risk_square(self):
        """
        Returns the location and probability of the most dangerous square.
        Useful for 'Trapdoor Judo' (baiting opponent).
        """
        max_white = np.max(self.belief_white)
        arg_white = np.unravel_index(np.argmax(self.belief_white), self.belief_white.shape)
        
        max_black = np.max(self.belief_black)
        arg_black = np.unravel_index(np.argmax(self.belief_black), self.belief_black.shape)
        
        if max_white > max_black:
            return arg_white, max_white
        else:
            return arg_black, max_black

    def debug_print(self):
        print("--- White Trapdoor Belief ---")
        print(np.round(self.belief_white, 3))
        print("\n--- Black Trapdoor Belief ---")
        print(np.round(self.belief_black, 3))