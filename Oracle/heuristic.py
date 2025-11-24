import math

class HeuristicEvaluator:
    def __init__(self, player_id):
        self.player_id = player_id
        # Weights for different components of the score
        self.W_SCORE = 1000.0      # The actual game score (eggs)
        self.W_MOBILITY = 300.0     # Number of squares we can reach vs opponent
        self.W_SAFETY = 10000.0     # Penalty for standing on a likely trap
        self.W_TURDS = 250.0        # Penalty for wasting turds early? Or bonus for saving them?
        
    def evaluate(self, state, belief_engine):
        if state.is_game_over:
            if state.winner == self.player_id:
                return float('inf')
            elif state.winner is not None:
                return float('-inf')
            else:
                return 0.0  # Draw

        my_score = state.scores[self.player_id]
        opp_score = state.scores[1 - self.player_id]
        score_diff = my_score - opp_score

        my_pos = state.positions[self.player_id]
        risk_prob = belief_engine.get_risk(*my_pos)
        
        risk_penalty = risk_prob * risk_prob * self.W_SAFETY

        my_mobility = self._calculate_mobility(state, self.player_id)
        opp_mobility = self._calculate_mobility(state, 1 - self.player_id)
        
        if my_mobility == 0:
            return float('-inf')
        if opp_mobility == 0:
            return float('inf') # We trapped them!

        mobility_diff = my_mobility - opp_mobility

        total_score = (
            (score_diff * self.W_SCORE) +
            (mobility_diff * self.W_MOBILITY) -
            risk_penalty
        )
        
        return total_score

    def _calculate_mobility(self, state, p_id):
        """
        Performs a BFS to count how many unique squares a player can reach.
        This approximates 'board control'.
        """
        start_pos = state.positions[p_id]
        queue = [start_pos]
        visited = {start_pos}
        count = 0
        
        max_depth = 10 
        depth = 0
        
        while queue and depth < max_depth:
            next_queue = []
            for curr in queue:
                count += 1
                neighbors = self._get_valid_neighbors(state, curr, p_id)
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        next_queue.append(n)
            queue = next_queue
            depth += 1
            
        return count

    def _get_valid_neighbors(self, state, pos, p_id):
        """
        Returns valid adjacent moves for mobility calculation.
        Note: This mimics the game rules for movement.
        """
        x, y = pos
        moves = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < state.width and 0 <= ny < state.height:
                if state.is_blocked(nx, ny, p_id):
                    continue
                moves.append((nx, ny))
        return moves