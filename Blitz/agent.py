"""
BLITZ - The Speed Demon Agent
Strategy: LAY EGGS AS FAST AS HUMANLY POSSIBLE
- Lay egg every turn if possible (zero safety checks)
- If can't lay egg, move to nearest empty square
- No planning, no thinking, no trap avoidance
- PURE SPEED
"""

from typing import Tuple, List, Optional
from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction

class PlayerAgent:
    def __init__(self, board: Board, time_left):
        """Initialize Blitz - the speed demon."""
        my_loc = board.chicken_player.get_location()
        self.my_parity = (my_loc[0] + my_loc[1]) % 2
        self.location_history = []  # Track recent locations to avoid oscillation
        
        
    def play(self, board: Board, sensor_data, time_left) -> Tuple[Direction, MoveType]:
        """
        PURE SPEED STRATEGY + ANTI-OSCILLATION:
        1. Lay egg if possible (ALWAYS)
        2. Move to nearest empty square (avoid recent locations)
        3. Don't waste turns going back and forth
        NO SAFETY CHECKS. FULL SEND.
        """
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return (Direction.UP, MoveType.PLAIN)
        
        my_loc = board.chicken_player.get_location()
        
        # PRIORITY 1: LAY EGG IMMEDIATELY (no safety check!)
        egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
        if egg_moves:
            # Prefer corners for 3x multiplier
            corner_moves = []
            for move in egg_moves:
                next_loc = loc_after_direction(my_loc, move[0])
                if next_loc in [(0,0), (0,7), (7,0), (7,7)]:
                    corner_moves.append(move)
            
            # Update history before returning
            self.location_history.append(my_loc)
            if len(self.location_history) > 8:
                self.location_history.pop(0)
            
            return corner_moves[0] if corner_moves else egg_moves[0]
        
        # PRIORITY 2: Move to nearest empty square (AVOID RECENT LOCATIONS)
        nearest = self._find_nearest_empty(board, my_loc)
        if nearest:
            # Simple greedy move toward target
            dx = nearest[0] - my_loc[0]
            dy = nearest[1] - my_loc[1]
            
            plain_moves = [m for m in valid_moves if m[1] == MoveType.PLAIN]
            
            # Filter out moves that return to recent locations (anti-oscillation)
            smart_moves = []
            for move in plain_moves:
                next_loc = loc_after_direction(my_loc, move[0])
                
                # CRITICAL: Don't immediately backtrack (A -> B -> A pattern)
                if len(self.location_history) >= 1 and next_loc == self.location_history[-1]:
                    continue  # Skip this move
                
                # Avoid recent history (last 4 locations)
                if next_loc in self.location_history[-4:]:
                    continue  # Skip this move
                
                smart_moves.append(move)
            
            # Use smart_moves if available, otherwise fall back to all plain moves
            moves_to_consider = smart_moves if smart_moves else plain_moves
            
            # Try horizontal first
            if abs(dx) >= abs(dy):
                preferred_dir = Direction.RIGHT if dx > 0 else Direction.LEFT
            else:
                preferred_dir = Direction.DOWN if dy > 0 else Direction.UP
            
            preferred_move = (preferred_dir, MoveType.PLAIN)
            if preferred_move in moves_to_consider:
                self.location_history.append(my_loc)
                if len(self.location_history) > 8:
                    self.location_history.pop(0)
                return preferred_move
            
            # Fallback: any smart move
            if moves_to_consider:
                self.location_history.append(my_loc)
                if len(self.location_history) > 8:
                    self.location_history.pop(0)
                return moves_to_consider[0]
        
        # PRIORITY 3: Just move anywhere (but avoid backtracking)
        plain_moves = [m for m in valid_moves if m[1] == MoveType.PLAIN]
        if plain_moves:
            # Try to avoid immediate backtrack
            for move in plain_moves:
                next_loc = loc_after_direction(my_loc, move[0])
                if len(self.location_history) >= 1 and next_loc == self.location_history[-1]:
                    continue
                
                self.location_history.append(my_loc)
                if len(self.location_history) > 8:
                    self.location_history.pop(0)
                return move
            
            # If all moves backtrack, just take the first one
            self.location_history.append(my_loc)
            if len(self.location_history) > 8:
                self.location_history.pop(0)
            return plain_moves[0]
        
        # LAST RESORT: Whatever is valid
        self.location_history.append(my_loc)
        if len(self.location_history) > 8:
            self.location_history.pop(0)
        return valid_moves[0]
    
    def _find_nearest_empty(self, board: Board, my_loc: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest square that doesn't have an egg (ours or theirs)."""
        size = board.game_map.MAP_SIZE
        best = None
        best_dist = 999
        
        for x in range(size):
            for y in range(size):
                loc = (x, y)
                
                # Check parity
                if (x + y) % 2 != self.my_parity:
                    continue
                
                # Skip if already has egg
                if loc in board.eggs_player or loc in board.eggs_enemy:
                    continue
                
                # Skip if has turd (at least avoid OBVIOUS blocks)
                if loc in board.turds_player or loc in board.turds_enemy:
                    continue
                
                dist = abs(my_loc[0] - x) + abs(my_loc[1] - y)
                if dist < best_dist:
                    best_dist = dist
                    best = loc
        
        return best
