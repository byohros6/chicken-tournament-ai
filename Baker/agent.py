import time
import queue
import numpy as np
from typing import List, Tuple, Callable
from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction
from .oracle import Oracle
from .search import SearchAgent

class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.oracle = Oracle(self.map_size)
        self.search_engine = SearchAgent()
        self.visited_corners = set()
        self.turn_number = 0
        
        # Parity Logic
        my_parity = board.chicken_player.even_chicken
        all_corners = [
            (0, 0), (0, self.map_size - 1), 
            (self.map_size - 1, 0), (self.map_size - 1, self.map_size - 1)
        ]
        self.corners = [c for c in all_corners if (c[0] + c[1]) % 2 == my_parity]
        
        self.visit_counts = {} 

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        self.turn_number += 1
        remaining_time = time_left()
        
        my_loc = board.chicken_player.get_location()
        self.oracle.update(my_loc, sensor_data)
        
        current_count = self.visit_counts.get(my_loc, 0)
        self.visit_counts[my_loc] = current_count + 1

        # Panic Mode
        if remaining_time < 2.0:
             return self.search_engine.get_best_move(board, self.oracle, time_left, self.visit_counts)

        # --- OPENING BOOK ---
        if self.turn_number < 40: 
            best_corner_move = self._run_opening_book(board)
            if best_corner_move:
                return best_corner_move
        
        # Combat Phase
        return self.search_engine.get_best_move(board, self.oracle, time_left, self.visit_counts)

    def _run_opening_book(self, board):
        my_loc = board.chicken_player.get_location()
        target_corner = None
        min_dist = 999
        
        for corner in self.corners:
            if corner in board.eggs_player:
                continue

            dist = abs(my_loc[0] - corner[0]) + abs(my_loc[1] - corner[1])
            if dist < min_dist:
                # FIX: Increased tolerance to 0.15 (15%). 
                # We must be brave to win the corners!
                if self.oracle.get_risk(corner) < 0.15:
                    target_corner = corner
                    min_dist = dist
                    
        if not target_corner: return None

        # 1. ARRIVAL: If at corner, lay egg
        if my_loc == target_corner:
            if board.can_lay_egg():
                for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                    if board.is_valid_move(d, MoveType.EGG):
                        return (d, MoveType.EGG)
            return None

        # 2. NAVIGATION
        path = self._astar_search(board, my_loc, target_corner)
        if not path: return None
            
        next_step = path[0]
        direction = self._get_direction(my_loc, next_step)
        
        # 3. HARVEST EN ROUTE (The "Greedy" Fix)
        # If the move allows us to lay an egg AND step in the right direction, do it.
        if board.can_lay_egg():
            if board.is_valid_move(direction, MoveType.EGG):
                return (direction, MoveType.EGG)
        
        return (direction, MoveType.PLAIN)

    def _astar_search(self, board, start, goal):
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
                
                risk = self.oracle.get_risk(next_loc)
                
                # FIX: Increased tolerance in pathfinding too.
                # If we don't risk it, we don't get the biscuit (corner).
                if risk > 0.15: continue
                    
                new_cost = cost_so_far[current] + 1
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
        return path

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_direction(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if dx == 1: return Direction.RIGHT
        if dx == -1: return Direction.LEFT
        if dy == 1: return Direction.DOWN
        return Direction.UP