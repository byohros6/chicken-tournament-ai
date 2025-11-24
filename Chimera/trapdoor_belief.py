"""
Enhanced Bayesian Trapdoor Belief Tracking
Uses log probabilities for numerical stability and proper decay
"""
import numpy as np
from typing import Tuple, List
from game.game_map import prob_feel, prob_hear


class TrapdoorBelief:
    """
    Maintains probability distributions over trapdoor locations using Bayesian inference.
    Enhanced with log probabilities and proper decay to avoid overconfidence.
    """
    
    def __init__(self, map_size: int = 8):
        self.map_size = map_size
        self.decay_rate = 0.03  # Slower decay than Gertrude (0.05) - more conservative
        
        # Log probabilities for numerical stability
        self.log_probs_even = None
        self.log_probs_odd = None
        
        # Track confirmed trapdoors (when actually triggered)
        self.confirmed_trapdoors = set()
        
        self._initialize_priors()
    
    def _initialize_priors(self) -> None:
        """
        Initialize prior probabilities based on distance from edge.
        Center squares more likely (as per trapdoor_manager.py rules).
        """
        weights = np.zeros((self.map_size, self.map_size), dtype=np.float64)
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                # Distance from nearest edge
                dist_x = min(x, self.map_size - 1 - x)
                dist_y = min(y, self.map_size - 1 - y)
                ring = min(dist_x, dist_y)
                
                # Weights match the game's trapdoor generation
                if ring >= 3:
                    weight = 2.0
                elif ring == 2:
                    weight = 1.0
                elif ring == 1:
                    weight = 0.2  # Small but non-zero
                else:  # ring == 0 (edge)
                    weight = 0.05  # Very unlikely but possible
                
                weights[x, y] = weight
        
        # Separate weights for even and odd parity
        even_weights = weights.copy()
        odd_weights = weights.copy()
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                if (x + y) % 2 != 0:
                    even_weights[x, y] = 0.0
                else:
                    odd_weights[x, y] = 0.0
        
        # Normalize and convert to log probabilities
        even_sum = np.sum(even_weights)
        odd_sum = np.sum(odd_weights)
        
        if even_sum > 0:
            even_probs = even_weights / even_sum
            self.log_probs_even = np.log(even_probs + 1e-10)  # Small epsilon to avoid log(0)
        else:
            self.log_probs_even = np.full((self.map_size, self.map_size), -np.inf)
        
        if odd_sum > 0:
            odd_probs = odd_weights / odd_sum
            self.log_probs_odd = np.log(odd_probs + 1e-10)
        else:
            self.log_probs_odd = np.full((self.map_size, self.map_size), -np.inf)
        
        # Store initial priors for decay
        self.log_priors_even = self.log_probs_even.copy()
        self.log_priors_odd = self.log_probs_odd.copy()
    
    def update(self, location: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]) -> None:
        """
        Update trapdoor beliefs using Bayesian inference with sensor data.
        
        Args:
            location: Current (x, y) position
            sensor_data: [(heard_even, felt_even), (heard_odd, felt_odd)]
        """
        # Apply decay: slowly drift back toward priors to avoid overconfidence
        self.log_probs_even = (1 - self.decay_rate) * self.log_probs_even + \
                              self.decay_rate * self.log_priors_even
        self.log_probs_odd = (1 - self.decay_rate) * self.log_probs_odd + \
                             self.decay_rate * self.log_priors_odd
        
        # Update even trapdoor beliefs
        heard_even, felt_even = sensor_data[0]
        self._bayesian_update_parity(location, heard_even, felt_even, parity=0)
        
        # Update odd trapdoor beliefs
        heard_odd, felt_odd = sensor_data[1]
        self._bayesian_update_parity(location, heard_odd, felt_odd, parity=1)
    
    def _bayesian_update_parity(self, location: Tuple[int, int], 
                                 heard: bool, felt: bool, parity: int) -> None:
        """
        Perform Bayesian update for one parity (even or odd).
        """
        log_probs = self.log_probs_even if parity == 0 else self.log_probs_odd
        
        # Compute log likelihoods
        log_likelihoods = np.full((self.map_size, self.map_size), -np.inf, dtype=np.float64)
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                if (x + y) % 2 != parity:
                    continue
                
                # Calculate distance
                dx = abs(x - location[0])
                dy = abs(y - location[1])
                
                # Get probabilities of hearing and feeling from this distance
                p_hear = prob_hear(dx, dy)
                p_feel = prob_feel(dx, dy)
                
                # Likelihood: P(sensor_data | trapdoor at (x,y))
                # Assuming independence: P(heard, felt | trapdoor) = P(heard | trapdoor) * P(felt | trapdoor)
                likelihood_hear = p_hear if heard else (1 - p_hear)
                likelihood_feel = p_feel if felt else (1 - p_feel)
                
                # Log likelihood
                log_likelihood = np.log(likelihood_hear + 1e-10) + np.log(likelihood_feel + 1e-10)
                log_likelihoods[x, y] = log_likelihood
        
        # Bayesian update: log posterior = log prior + log likelihood
        log_posteriors = log_probs + log_likelihoods
        
        # Normalize in log space (log-sum-exp trick for numerical stability)
        max_log_posterior = np.max(log_posteriors[log_posteriors > -np.inf])
        if max_log_posterior == -np.inf:
            return  # No valid probabilities
        
        # Convert to normal space for normalization
        posteriors = np.exp(log_posteriors - max_log_posterior)
        posterior_sum = np.sum(posteriors)
        
        if posterior_sum > 0:
            posteriors /= posterior_sum
            # Convert back to log space
            if parity == 0:
                self.log_probs_even = np.log(posteriors + 1e-10)
            else:
                self.log_probs_odd = np.log(posteriors + 1e-10)
    
    def get_prob_at(self, location: Tuple[int, int]) -> float:
        """
        Get probability that a trapdoor is at this location.
        """
        x, y = location
        parity = (x + y) % 2
        
        if parity == 0:
            log_prob = self.log_probs_even[x, y]
        else:
            log_prob = self.log_probs_odd[x, y]
        
        return np.exp(log_prob)
    
    def mark_confirmed_trapdoor(self, location: Tuple[int, int]) -> None:
        """
        Mark a location as a confirmed trapdoor (100% certainty).
        """
        self.confirmed_trapdoors.add(location)
        x, y = location
        parity = (x + y) % 2
        
        # Set this location to 100% and all others to 0%
        if parity == 0:
            self.log_probs_even = np.full((self.map_size, self.map_size), -np.inf)
            self.log_probs_even[x, y] = 0.0  # log(1) = 0
        else:
            self.log_probs_odd = np.full((self.map_size, self.map_size), -np.inf)
            self.log_probs_odd[x, y] = 0.0
    
    def is_confirmed_trapdoor(self, location: Tuple[int, int]) -> bool:
        """Check if location is a confirmed trapdoor."""
        return location in self.confirmed_trapdoors
    
    def get_high_risk_locations(self, threshold: float = 0.15) -> List[Tuple[int, int]]:
        """
        Get list of locations with trapdoor probability above threshold.
        """
        risky = []
        for x in range(self.map_size):
            for y in range(self.map_size):
                prob = self.get_prob_at((x, y))
                if prob > threshold:
                    risky.append((x, y))
        return risky
