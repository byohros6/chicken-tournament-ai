# agent.py - Competitive Chicken Bot
# Strategy: Minimax with Alpha-Beta Pruning + Trapdoor Inference + Aggressive Egg Laying


from typing import Tuple, List, Set, Optional
from collections import defaultdict
import random

from game.enums import Direction, MoveType
from game.board import Board



class TrapdoorTracker:
    """Track and infer trapdoor locations based on sensory data."""
    
    def __init__(self):
        # Probability map for each trapdoor (white=even, black=odd squares)
        self.white_probs = {}  # (x, y) -> probability
        self.black_probs = {}
        
        # Initialize all valid squares with base probability
        for x in range(8):
            for y in range(8):
                if (x + y) % 2 == 0:  # White square
                    self.white_probs[(x, y)] = 1.0 / 32  # 32 white squares
                else:  # Black square
                    self.black_probs[(x, y)] = 1.0 / 32
    
    def update(self, position: Tuple[int, int], heard_white: bool, felt_white: bool, 
               heard_black: bool, felt_black: bool):
        """Update trapdoor probabilities based on sensory data."""
        x, y = position
        
        # Update white trapdoor probabilities
        self._update_trapdoor(self.white_probs, position, heard_white, felt_white, is_even=True)
        
        # Update black trapdoor probabilities
        self._update_trapdoor(self.black_probs, position, heard_black, felt_black, is_even=False)
    
    def _update_trapdoor(self, prob_map, position, heard, felt, is_even):
        """Bayesian update for a single trapdoor."""
        x, y = position
        
        # Define proximity zones
        adjacent = []  # Shares edge
        diagonal = []  # Diagonal
        nearby = []    # Two squares away
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if (nx + ny) % 2 == (0 if is_even else 1):
                        if abs(dx) + abs(dy) == 1:
                            adjacent.append((nx, ny))
                        else:
                            diagonal.append((nx, ny))
        
        # Two squares away (for 10% hearing chance)
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), 
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if (nx + ny) % 2 == (0 if is_even else 1):
                    if (nx, ny) not in adjacent and (nx, ny) not in diagonal:
                        # Check if adjacent to adjacent/diagonal squares
                        for ax, ay in adjacent + diagonal:
                            if abs(nx - ax) + abs(ny - ay) == 1:
                                nearby.append((nx, ny))
                                break
        
        # Likelihood multipliers based on sensory data
        for pos, prob in list(prob_map.items()):
            if pos in adjacent:
                # 50% hear, 30% feel
                likelihood = 1.0
                if heard:
                    likelihood *= 0.5
                else:
                    likelihood *= 0.5
                if felt:
                    likelihood *= 0.3
                else:
                    likelihood *= 0.7
            elif pos in diagonal:
                # 25% hear, 15% feel
                likelihood = 1.0
                if heard:
                    likelihood *= 0.25
                else:
                    likelihood *= 0.75
                if felt:
                    likelihood *= 0.15
                else:
                    likelihood *= 0.85
            elif pos in nearby:
                # 10% hear, 0% feel
                likelihood = 1.0
                if heard:
                    likelihood *= 0.1
                else:
                    likelihood *= 0.9
                if felt:
                    likelihood *= 1.0  # Can't feel from here
            else:
                # No sensory data from here
                if heard or felt:
                    likelihood = 0.01  # Very unlikely
                else:
                    likelihood = 1.0
            
            prob_map[pos] = prob * likelihood
        
        # Normalize probabilities
        total = sum(prob_map.values())
        if total > 0:
            for pos in prob_map:
                prob_map[pos] /= total
    
    def get_danger_score(self, position: Tuple[int, int]) -> float:
        """Get probability that this position is a trapdoor."""
        x, y = position
        if (x + y) % 2 == 0:
            return self.white_probs.get(position, 0.0)
        else:
            return self.black_probs.get(position, 0.0)
    
    def get_most_likely_trapdoors(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return most likely positions for both trapdoors."""
        white_pos = max(self.white_probs.items(), key=lambda x: x[1])[0]
        black_pos = max(self.black_probs.items(), key=lambda x: x[1])[0]
        return white_pos, black_pos


class PlayerAgent:
    """Competitive Chicken Agent using Minimax with Alpha-Beta Pruning."""
    
    def __init__(self, board: Board, time_left):
        """Constructor signature expected by the engine.

        `board` is the initial Board; `time_left` is a callable returning
        remaining time. We don't need them at construction time but must
        accept them to avoid a constructor TypeError.
        """

        self.trapdoor_tracker = TrapdoorTracker()
        self.history = []
        self.move_count = 0

    def play(self, board: Board, trapdoor_data: List[Tuple[bool, bool]], time_left: callable) -> Tuple[Direction, MoveType]:
        """Main decision function called each turn."""
        self.move_count += 1
        
        # Update trapdoor knowledge
        heard_white, felt_white = trapdoor_data[0]
        heard_black, felt_black = trapdoor_data[1]
        current_pos = board.chicken_player.get_location()
        self.trapdoor_tracker.update(current_pos, heard_white, felt_white, heard_black, felt_black)
        
        # Get valid moves
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            # Should never happen, but return a random direction/type
            return (Direction.UP, MoveType.PLAIN)
        
        # Time management: Adjust search depth based on time remaining
        time_remaining = time_left()
        if time_remaining > 240:  # More than 4 minutes
            max_depth = 4
        elif time_remaining > 120:  # More than 2 minutes
            max_depth = 3
        elif time_remaining > 30:  # More than 30 seconds
            max_depth = 2
        else:
            max_depth = 1
        
        # Use minimax to find best move
        best_move = self.minimax_decision(board, valid_moves, max_depth, time_left)
        
        return best_move
    
    def minimax_decision(self, board: Board, valid_moves: List[Tuple[Direction, MoveType]], 
                         max_depth: int, time_left: callable) -> Tuple[Direction, MoveType]:
        """Find best move using Minimax with Alpha-Beta Pruning."""
        best_value = float('-inf')
        best_move = valid_moves[0]
        alpha = float('-inf')
        beta = float('inf')
        
        # Shuffle to avoid deterministic play
        random.shuffle(valid_moves)
        
        for move in valid_moves:
            # Check time constraint
            if time_left() < 1.0:
                break
            
            # Simulate move
            new_board = board.forecast_move(move[0], move[1])
            
            # Evaluate
            value = self.min_value(new_board, alpha, beta, max_depth - 1, time_left)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)
        
        return best_move
    
    def min_value(self, board: Board, alpha: float, beta: float, depth: int, time_left: callable) -> float:
        """Minimax MIN node (opponent's turn)."""
        if depth == 0 or time_left() < 0.5:
            return self.evaluate(board)
        
        # Reverse perspective in-place to get opponent's valid moves
        board.reverse_perspective()
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            # Opponent has no moves - we win!
            # Restore perspective before returning
            board.reverse_perspective()
            return float('inf')
        
        value = float('inf')
        for move in valid_moves[:10]:  # Limit branching factor
            # Simulate opponent move from their perspective
            new_board = board.forecast_move(move[0], move[1])
            # Reverse back to our perspective for the recursive call
            new_board.reverse_perspective()

            value = min(value, self.max_value(new_board, alpha, beta, depth - 1, time_left))
            
            if value <= alpha:
                # Restore perspective before cutoff
                board.reverse_perspective()
                return value  # Alpha cutoff
            beta = min(beta, value)
        
        # Restore perspective before returning
        board.reverse_perspective()
        return value
    
    def max_value(self, board: Board, alpha: float, beta: float, depth: int, time_left: callable) -> float:
        """Minimax MAX node (our turn)."""
        if depth == 0 or time_left() < 0.5:
            return self.evaluate(board)
        
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            # We have no moves - we lose
            return float('-inf')
        
        value = float('-inf')
        for move in valid_moves[:10]:  # Limit branching factor
            new_board = board.forecast_move(move[0], move[1])
            
            value = max(value, self.min_value(new_board, alpha, beta, depth - 1, time_left))
            
            if value >= beta:
                return value  # Beta cutoff
            alpha = max(alpha, value)
        
        return value
    
    def evaluate(self, board: Board) -> float:
        """Heuristic evaluation function."""
        score = 0.0
        
        # 1. Egg difference (most important)
        egg_diff = len(board.eggs_player) - len(board.eggs_enemy)
        score += egg_diff * 100
        
        # 2. Control of corner squares (worth 3 eggs each)
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for corner in corners:
            if corner in board.eggs_player:
                score += 50
            if corner in board.eggs_enemy:
                score -= 50
        
        # 3. Mobility (number of valid moves)
        our_moves = len(board.get_valid_moves())
        score += our_moves * 5

        # Estimate opponent mobility without mutating the original board.
        # reverse_perspective() is in-place and returns None, so we work on
        # a copied board to preserve the evaluation logic.
        try:
            opp_board = board.copy()
        except AttributeError:
            # Fallback: if Board has no copy(), approximate with our moves.
            opp_board = None

        if opp_board is not None:
            opp_board.reverse_perspective()
            opp_moves = len(opp_board.get_valid_moves())
        else:
            opp_moves = our_moves

        score -= opp_moves * 5

        # 4. Blocking potential (turds near opponent)
        opp_pos = board.chicken_enemy.get_location()
        for turd_pos in board.turds_player:
            dist = abs(turd_pos[0] - opp_pos[0]) + abs(turd_pos[1] - opp_pos[1])
            if dist <= 2:
                score += 10

        # 5. Territory control (squares we can reach easily)
        our_pos = board.chicken_player.get_location()
        reachable = self.count_reachable_squares(board, our_pos, 3)
        score += reachable * 2

        # 6. Trapdoor avoidance
        danger = self.trapdoor_tracker.get_danger_score(our_pos)
        score -= danger * 200  # Heavy penalty for likely trapdoor

        # 7. Remaining turds (keep some for endgame) – not directly exposed, skip.
        # score += board.player.remaining_turds * 3

        # 8. Distance from opponent (stay away to avoid being blocked)
        dist_to_opp = abs(our_pos[0] - opp_pos[0]) + abs(our_pos[1] - opp_pos[1])
        if dist_to_opp < 3:
            score -= 10

        return score
    
    def count_reachable_squares(self, board: Board, start: Tuple[int, int], max_dist: int) -> int:
        """Count squares reachable within max_dist moves."""
        visited = set()
        queue = [(start, 0)]
        visited.add(start)
        
        while queue:
            pos, dist = queue.pop(0)
            if dist >= max_dist:
                continue
            
            x, y = pos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    npos = (nx, ny)
                    if npos not in visited:
                        # Check if we can move there
                        if (npos not in board.eggs_enemy and 
                            npos not in board.turds_enemy):
                            # Check adjacent turds
                            can_move = True
                            for tx, ty in board.turds_enemy:
                                if abs(nx - tx) + abs(ny - ty) == 1:
                                    can_move = False
                                    break
                            if can_move:
                                visited.add(npos)
                                queue.append((npos, dist + 1))
        
        return len(visited)
