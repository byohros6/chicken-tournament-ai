"""
Strategic Planning Module
Handles turd placement optimization and game phase management
"""
from typing import List, Tuple, Set, Dict, Optional
from game.board import Board
from game.enums import Direction, loc_after_direction


class StrategicPlanner:
    """
    Manages high-level strategic decisions:
    - When to lay turds
    - Where to place turds for maximum opponent disruption
    - Game phase detection (early/mid/endgame)
    """
    
    def __init__(self, board: Board):
        self.map_size = board.game_map.MAP_SIZE
        self.my_parity = board.chicken_player.even_chicken
        self.enemy_parity = 1 - self.my_parity
        self.detected_barrier = False
        self.barrier_diagonal = None  # (dx, dy) direction of barrier
    
    def detect_barrier_strategy(self, board: Board) -> bool:
        """
        Detect if enemy is building a diagonal barrier wall.
        Returns True if 3+ enemy turds form a diagonal pattern.
        """
        if len(board.turds_enemy) < 3:
            return False
        
        # Check for diagonal pattern
        turds_list = list(board.turds_enemy)
        if len(turds_list) < 3:
            return False
        
        # Sort by position to find diagonal
        turds_sorted = sorted(turds_list)
        
        # Check if first 3 turds form diagonal (both x and y increasing/decreasing by 1)
        for i in range(len(turds_sorted) - 2):
            t1, t2, t3 = turds_sorted[i], turds_sorted[i+1], turds_sorted[i+2]
            dx1 = t2[0] - t1[0]
            dy1 = t2[1] - t1[1]
            dx2 = t3[0] - t2[0]
            dy2 = t3[1] - t2[1]
            
            # Diagonal: both dx and dy are +1 or -1, and consistent
            if abs(dx1) == 1 and abs(dy1) == 1 and dx1 == dx2 and dy1 == dy2:
                self.detected_barrier = True
                self.barrier_diagonal = (dx1, dy1)
                return True
        
        return False
    
    def get_game_phase(self, board: Board) -> str:
        """
        Determine current game phase.
        
        Returns:
            'early' (turns 1-12): Focus on egg collection
            'mid' (turns 13-28): Strategic turd placement
            'endgame' (turns 29-40): Final egg rush and denial
        """
        turn = board.turn_count
        
        if turn <= 12:
            return 'early'
        elif turn <= 28:
            return 'mid'
        else:
            return 'endgame'
    
    def should_lay_turd_now(self, board: Board, trapdoor_risks: Dict[Tuple[int, int], float]) -> bool:
        """
        Decide if we should lay a turd this turn.
        
        Strategy:
        - Early game: NO (collect eggs)
        - Mid game: YES if enemy threatens high-value squares
        - Endgame: YES if can deny critical eggs
        """
        phase = self.get_game_phase(board)
        
        if phase == 'early':
            return False  # Never lay turds early
        
        if board.chicken_player.get_turds_left() == 0:
            return False  # No turds left
        
        # Check if we can lay turd at current location
        my_loc = board.chicken_player.get_location()
        if not board.can_lay_turd_at_loc(my_loc):
            return False
        
        # Mid game: check if enemy is threatening valuable squares
        if phase == 'mid':
            enemy_threats = self._calculate_enemy_threats(board)
            # Lay turd if enemy is within 3 moves of a high-value square
            for threat_loc, threat_level in enemy_threats.items():
                if threat_level >= 3.0 and self._is_high_value_square(threat_loc):
                    # Check if laying turd here blocks that threat
                    if self._would_block_square(my_loc, threat_loc):
                        return True
        
        # Endgame: aggressive denial
        if phase == 'endgame':
            egg_diff = board.chicken_player.get_eggs_laid() - board.chicken_enemy.get_eggs_laid()
            
            # If we're ahead, block enemy's best squares
            if egg_diff > 0:
                enemy_loc = board.chicken_enemy.get_location()
                # Check if enemy is close to a valuable square
                for x in range(self.map_size):
                    for y in range(self.map_size):
                        loc = (x, y)
                        if (x + y) % 2 == self.enemy_parity:
                            if loc not in board.eggs_enemy and self._is_high_value_square(loc):
                                dist = abs(enemy_loc[0] - x) + abs(enemy_loc[1] - y)
                                if dist <= 2 and self._would_block_square(my_loc, loc):
                                    return True
            
            # If we're behind, still try to deny but prioritize our own eggs
            elif egg_diff < -2:
                return False  # Focus on our eggs when behind
        
        return False
    
    def find_best_turd_location(self, board: Board,
                               trapdoor_risks: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        """
        Find the best location to place a turd.
        
        Strategy:
        - Maximize enemy egg squares blocked
        - Minimize our own egg squares blocked
        - Consider trapdoor risk
        """
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        
        # Can't lay turd adjacent to enemy
        if abs(my_loc[0] - enemy_loc[0]) + abs(my_loc[1] - enemy_loc[1]) <= 1:
            return None
        
        # Current location (where turd would be placed)
        if not board.can_lay_turd_at_loc(my_loc):
            return None
        
        # Calculate impact of placing turd at current location
        impact_score = self._calculate_turd_impact(my_loc, board, trapdoor_risks)
        
        # Only place if impact is positive (helps us more than hurts)
        if impact_score > 2.0:  # Threshold for worthwhile turd
            return my_loc
        
        return None
    
    def _calculate_turd_impact(self, turd_loc: Tuple[int, int], board: Board,
                              trapdoor_risks: Dict[Tuple[int, int], float]) -> float:
        """
        Calculate the strategic value of placing a turd at this location.
        
        Returns positive score if beneficial, negative if harmful to us.
        """
        score = 0.0
        
        # A turd blocks 5 squares: itself + 4 adjacent
        blocked_squares = [turd_loc]
        for direction in Direction:
            adj = loc_after_direction(turd_loc, direction)
            if board.is_valid_cell(adj):
                blocked_squares.append(adj)
        
        # Evaluate each blocked square
        for sq in blocked_squares:
            x, y = sq
            sq_parity = (x + y) % 2
            
            # Does this block an enemy egg square?
            if sq_parity == self.enemy_parity:
                if sq not in board.eggs_enemy:  # Enemy hasn't laid there yet
                    value = self._square_value(sq)
                    score += value * 2.0  # Blocking enemy is good
            
            # Does this block our own egg square?
            elif sq_parity == self.my_parity:
                if sq not in board.eggs_player:  # We haven't laid there yet
                    value = self._square_value(sq)
                    score -= value * 1.5  # Blocking ourselves is bad
        
        # Penalize if turd is on high-risk trapdoor square
        trap_risk = trapdoor_risks.get(turd_loc, 0.0)
        score -= trap_risk * 10.0
        
        return score
    
    def _square_value(self, loc: Tuple[int, int]) -> float:
        """
        Calculate intrinsic value of a square for egg laying.
        Corners are worth more (3 eggs vs 1).
        """
        x, y = loc
        
        # Corners worth 3x
        if (x == 0 or x == self.map_size - 1) and (y == 0 or y == self.map_size - 1):
            return 3.0
        
        # Center squares worth more (easier to reach, more options)
        center_x = self.map_size / 2.0
        center_y = self.map_size / 2.0
        dist_from_center = abs(x - center_x) + abs(y - center_y)
        
        # Closer to center = more valuable (max 1.5, min 0.5)
        value = 1.5 - (dist_from_center / (self.map_size * 2)) * 1.0
        return max(0.5, value)
    
    def _is_high_value_square(self, loc: Tuple[int, int]) -> bool:
        """Check if square is high value (corner or center)."""
        x, y = loc
        
        # Corners
        if (x == 0 or x == self.map_size - 1) and (y == 0 or y == self.map_size - 1):
            return True
        
        # Center 4x4 area
        if 2 <= x <= 5 and 2 <= y <= 5:
            return True
        
        return False
    
    def _would_block_square(self, turd_loc: Tuple[int, int], target_square: Tuple[int, int]) -> bool:
        """Check if placing turd at turd_loc would block access to target_square."""
        # Turd blocks the square itself + adjacent squares (can't move through turd zone)
        if turd_loc == target_square:
            return True
        
        # Check if target is adjacent to turd
        dist = abs(turd_loc[0] - target_square[0]) + abs(turd_loc[1] - target_square[1])
        return dist == 1
    
    def _calculate_enemy_threats(self, board: Board) -> Dict[Tuple[int, int], float]:
        """
        Calculate which squares the enemy is threatening.
        Returns dict: location -> threat_level (higher = more imminent threat)
        """
        threats = {}
        enemy_loc = board.chicken_enemy.get_location()
        
        # Check all enemy parity squares
        for x in range(self.map_size):
            for y in range(self.map_size):
                loc = (x, y)
                if (x + y) % 2 == self.enemy_parity:
                    # Already has egg there?
                    if loc in board.eggs_enemy:
                        continue
                    
                    # Calculate threat level based on distance
                    dist = abs(enemy_loc[0] - x) + abs(enemy_loc[1] - y)
                    
                    # Closer = higher threat
                    if dist == 0:
                        threat_level = 10.0  # Enemy is ON the square
                    elif dist <= 2:
                        threat_level = 5.0
                    elif dist <= 4:
                        threat_level = 3.0
                    elif dist <= 6:
                        threat_level = 1.0
                    else:
                        threat_level = 0.5
                    
                    # Boost threat if it's a valuable square
                    if self._is_high_value_square(loc):
                        threat_level *= 1.5
                    
                    threats[loc] = threat_level
        
        return threats
    
    def get_strategic_target(self, board: Board, egg_tour: List[Tuple[int, int]],
                           trapdoor_risks: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        """
        Get the next strategic target to move toward.
        
        Priority:
        1. Next square in egg tour (if we're on a square we can lay an egg)
        2. Nearest unvisited high-value square
        3. Any unvisited egg square
        """
        my_loc = board.chicken_player.get_location()
        
        # If we have an egg tour, follow it
        if egg_tour:
            # Find first unvisited square in tour
            for target in egg_tour:
                if target not in board.eggs_player and target != my_loc:
                    # Make sure it's still accessible
                    if target not in board.eggs_enemy and target not in board.turds_enemy:
                        return target
        
        # Otherwise, find nearest high-value unvisited square
        best_target = None
        best_score = -float('inf')
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                loc = (x, y)
                
                # Only our parity squares
                if (x + y) % 2 != self.my_parity:
                    continue
                
                # Skip if already has egg or is blocked
                if loc in board.eggs_player or loc in board.eggs_enemy or loc in board.turds_enemy:
                    continue
                
                # Calculate score
                value = self._square_value(loc)
                dist = abs(my_loc[0] - x) + abs(my_loc[1] - y)
                trap_risk = trapdoor_risks.get(loc, 0.0)
                
                # Score: value / distance - risk
                score = value / (dist + 1.0) - trap_risk * 5.0
                
                if score > best_score:
                    best_score = score
                    best_target = loc
        
        return best_target
