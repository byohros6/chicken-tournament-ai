import queue
import random
from typing import List, Tuple, Callable
from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction
from .oracle import Oracle

class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.oracle = Oracle(self.map_size)
        self.turn_number = 0
        self.my_parity = board.chicken_player.even_chicken
        
        # 1. LAWNMOWER TOUR
        self.tour_targets = []
        for col in range(self.map_size):
            if col % 2 == 0:
                rows = range(self.map_size) # Down
            else:
                rows = range(self.map_size - 1, -1, -1) # Up
                
            for row in rows:
                if (row + col) % 2 == self.my_parity:
                    self.tour_targets.append((row, col))

        self.path_to_target = []
        self.visit_counts = {}
        
        # 2. DEATH TRAP MEMORY
        self.known_traps = set()
        self.start_pos = None
        self.last_target_move = None # Where did we TRY to go last turn?

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        self.turn_number += 1
        my_loc = board.chicken_player.get_location()
        self.oracle.update(my_loc, sensor_data)
        self.visit_counts[my_loc] = self.visit_counts.get(my_loc, 0) + 1
        
        # --- CRASH DETECTION LOGIC ---
        # If this is turn 1, memorize our spawn point
        if self.turn_number == 1:
            self.start_pos = my_loc
            
        # Did we die? 
        # If we are at start_pos, but we tried to go somewhere else... we fell in a hole.
        if self.last_target_move and my_loc == self.start_pos:
            if self.last_target_move != self.start_pos:
                # We tried to leave (or move elsewhere) but got sent back.
                # The square we tried to step ONTO is the trap.
                self.known_traps.add(self.last_target_move)
                # Force re-plan
                self.path_to_target = []

        # 1. HARVEST
        if self._can_score_here(board, my_loc):
            # If we are on a good spot, lay egg.
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                if board.is_valid_move(d, MoveType.EGG):
                    return self._record_move(my_loc, d, MoveType.EGG)

        # 2. TARGETING
        target = self._get_next_tour_target(board, my_loc)
        
        # 3. NAVIGATION
        if target:
            # Re-plan if needed or if blocked
            if not self.path_to_target or self.path_to_target[0] != my_loc:
                self.path_to_target = self._astar_path(board, my_loc, target)
            
            # Execute path
            if self.path_to_target and len(self.path_to_target) > 1:
                next_step = self.path_to_target[1]
                d = self._get_direction(my_loc, next_step)
                
                # Greedy Harvest En Route
                if board.can_lay_egg() and board.is_valid_move(d, MoveType.EGG):
                     return self._record_move(my_loc, d, MoveType.EGG)
                    
                if board.is_valid_move(d, MoveType.PLAIN):
                    self.path_to_target.pop(0)
                    return self._record_move(my_loc, d, MoveType.PLAIN)
                else:
                    self.path_to_target = [] # Blocked

        # 4. SURVIVAL
        return self._get_survival_move(board, my_loc)

    def _record_move(self, my_loc, direction, move_type):
        """Helper to save the intended destination for trap detection."""
        if move_type == MoveType.PLAIN:
            self.last_target_move = loc_after_direction(my_loc, direction)
        else:
            # Egg/Turd moves keep us in the same spot (conceptually for trap logic, 
            # though technically we step out. If we step onto a trap while laying, 
            # we handle that too).
            self.last_target_move = loc_after_direction(my_loc, direction)
            
        return (direction, move_type)

    def _get_next_tour_target(self, board, my_loc):
        best_target = None
        min_dist = 999
        
        for t in self.tour_targets:
            # Skip occupied
            if t in board.eggs_player: continue
            if t in board.turds_player: continue
            if t in board.turds_enemy: continue
            if board.is_cell_in_enemy_turd_zone(t): continue
            
            # Skip Known Death Traps (The Fix)
            if t in self.known_traps: continue

            # Skip High Risk (Oracle)
            if self.oracle.get_risk(t) > 0.20: continue
            
            dist = abs(t[0] - my_loc[0]) + abs(t[1] - my_loc[1])
            if dist < min_dist:
                min_dist = dist
                best_target = t
                if dist <= 3: return t
                    
        return best_target

    def _astar_path(self, board, start, goal):
        frontier = queue.PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while not frontier.empty():
            _, current = frontier.get()
            if current == goal: break
            
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                next_loc = loc_after_direction(current, d)
                
                if not (0 <= next_loc[0] < self.map_size and 0 <= next_loc[1] < self.map_size): continue
                if board.is_cell_blocked(next_loc): continue
                
                # AVOID TRAPS
                if next_loc in self.known_traps: continue
                if self.oracle.get_risk(next_loc) > 0.20: continue
                
                # Visits Penalty
                visits = self.visit_counts.get(next_loc, 0)
                step_cost = 1 + (visits * 50) 
                
                new_cost = cost_so_far[current] + step_cost
                if next_loc not in cost_so_far or new_cost < cost_so_far[next_loc]:
                    cost_so_far[next_loc] = new_cost
                    priority = new_cost + self._heuristic(next_loc, goal)
                    frontier.put((priority, next_loc))
                    came_from[next_loc] = current
                    
        if goal not in came_from: return None
        
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        path.insert(0, start)
        return path

    def _get_survival_move(self, board, my_loc):
        valid_moves = board.get_valid_moves()
        random.shuffle(valid_moves)
        best_move = (Direction.UP, MoveType.PLAIN)
        min_visits = float('inf')
        
        for move in valid_moves:
            if move[1] == MoveType.PLAIN:
                next_pos = loc_after_direction(my_loc, move[0])
                if next_pos in self.known_traps: continue # Don't walk into known death
                if self.oracle.get_risk(next_pos) > 0.50: continue
                
                visits = self.visit_counts.get(next_pos, 0)
                if visits < min_visits:
                    min_visits = visits
                    best_move = move
        
        # Record the survival move too, just in case it kills us
        return self._record_move(my_loc, best_move[0], best_move[1])

    def _can_score_here(self, board, loc):
        if (loc[0] + loc[1]) % 2 != self.my_parity: return False
        if loc in board.eggs_player: return False
        if loc in board.turds_player: return False
        if loc in board.turds_enemy: return False
        if loc in board.eggs_enemy: return False
        return True

    def _get_direction(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if dx == 1: return Direction.RIGHT
        if dx == -1: return Direction.LEFT
        if dy == 1: return Direction.DOWN
        return Direction.UP

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])