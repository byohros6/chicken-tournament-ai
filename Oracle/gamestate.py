import copy

class GameState:
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        
        # Player 0 (A) info
        self.p0_pos = None
        self.p0_score = 0
        self.p0_turds_left = 0
        self.p0_eggs = set()
        self.p0_turds = set()

        # Player 1 (B) info
        self.p1_pos = None
        self.p1_score = 0
        self.p1_turds_left = 0
        self.p1_eggs = set()
        self.p1_turds = set()

        self.turn_count = 0
        self.is_game_over = False
        self.winner = None
        
        # Indexed access
        self.positions = [None, None]
        self.scores = [0, 0]
        self.turds_left = [0, 0]
        self.eggs = [set(), set()]
        self.turds = [set(), set()]

    @classmethod
    def from_engine_board(cls, board, my_id):
        """
        Ingests state from the game engine.
        my_id=0 means we are Player A (White).
        my_id=1 means we are Player B (Black).
        """
        gs = cls(8, 8)
        
        # In the engine:
        # board.chicken_player is ME
        # board.chicken_enemy is OPPONENT
        me_chk = board.chicken_player
        opp_chk = board.chicken_enemy
        
        if my_id == 0:
            # I am P0
            p0_chk, p1_chk = me_chk, opp_chk
            p0_eggs_src, p1_eggs_src = board.eggs_player, board.eggs_enemy
            p0_turds_src, p1_turds_src = board.turds_player, board.turds_enemy
        else:
            # I am P1
            p1_chk, p0_chk = me_chk, opp_chk
            p1_eggs_src, p0_eggs_src = board.eggs_player, board.eggs_enemy
            p1_turds_src, p0_turds_src = board.turds_player, board.turds_enemy

        # Populate P0 data
        gs.p0_pos = p0_chk.get_location()
        gs.p0_score = p0_chk.get_eggs_laid()
        gs.p0_turds_left = p0_chk.get_turds_left()
        gs.p0_eggs = set(p0_eggs_src)
        gs.p0_turds = set(p0_turds_src)

        # Populate P1 data
        gs.p1_pos = p1_chk.get_location()
        gs.p1_score = p1_chk.get_eggs_laid()
        gs.p1_turds_left = p1_chk.get_turds_left()
        gs.p1_eggs = set(p1_eggs_src)
        gs.p1_turds = set(p1_turds_src)

        # Link to lists
        gs.positions = [gs.p0_pos, gs.p1_pos]
        gs.scores = [gs.p0_score, gs.p1_score]
        gs.turds_left = [gs.p0_turds_left, gs.p1_turds_left]
        gs.eggs = [gs.p0_eggs, gs.p1_eggs]
        gs.turds = [gs.p0_turds, gs.p1_turds]

        gs.turn_count = board.turn_count
        return gs

    def clone(self):
        new_state = GameState(self.width, self.height)
        new_state.p0_pos = self.p0_pos
        new_state.p0_score = self.p0_score
        new_state.p0_turds_left = self.p0_turds_left
        new_state.p0_eggs = self.p0_eggs.copy()
        new_state.p0_turds = self.p0_turds.copy()

        new_state.p1_pos = self.p1_pos
        new_state.p1_score = self.p1_score
        new_state.p1_turds_left = self.p1_turds_left
        new_state.p1_eggs = self.p1_eggs.copy()
        new_state.p1_turds = self.p1_turds.copy()

        new_state.positions = [new_state.p0_pos, new_state.p1_pos]
        new_state.scores = [new_state.p0_score, new_state.p1_score]
        new_state.turds_left = [new_state.p0_turds_left, new_state.p1_turds_left]
        new_state.eggs = [new_state.p0_eggs, new_state.p1_eggs]
        new_state.turds = [new_state.p0_turds, new_state.p1_turds]
        
        new_state.turn_count = self.turn_count
        return new_state

    def get_legal_actions(self, player_id):
        if self.is_game_over:
            return []

        actions = []
        curr_x, curr_y = self.positions[player_id]
        opp_id = 1 - player_id
        
        directions = [("UP", 0, -1), ("DOWN", 0, 1), ("LEFT", -1, 0), ("RIGHT", 1, 0)]

        for d_name, dx, dy in directions:
            nx, ny = curr_x + dx, curr_y + dy
            
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue

            if (nx, ny) == self.positions[opp_id]: continue
            if (nx, ny) in self.eggs[opp_id]: continue
            if (nx, ny) in self.turds[opp_id]: continue

            is_near_turd = False
            for tx, ty in [(0,1), (0,-1), (1,0), (-1,0)]:
                if (nx + tx, ny + ty) in self.turds[opp_id]:
                    is_near_turd = True
                    break
            if is_near_turd: continue

            actions.append((d_name, "PLAIN"))

            is_occupied = ((curr_x, curr_y) in self.eggs[player_id] or 
                           (curr_x, curr_y) in self.eggs[opp_id] or
                           (curr_x, curr_y) in self.turds[player_id] or 
                           (curr_x, curr_y) in self.turds[opp_id])
            
            parity_match = ((curr_x + curr_y) % 2 == 0) if player_id == 0 else ((curr_x + curr_y) % 2 != 0)
            
            if not is_occupied and parity_match:
                actions.append((d_name, "EGG"))

            if self.turds_left[player_id] > 0 and not is_occupied:
                opp_x, opp_y = self.positions[opp_id]
                dist = abs(curr_x - opp_x) + abs(curr_y - opp_y)
                if dist > 1: 
                    actions.append((d_name, "TURD"))
        
        return actions

    def apply_action(self, player_id, action):
        new_gs = self.clone()
        d_name, m_type = action
        dx, dy = 0, 0
        if d_name == "UP": dy = -1
        elif d_name == "DOWN": dy = 1
        elif d_name == "LEFT": dx = -1
        elif d_name == "RIGHT": dx = 1
        
        curr_x, curr_y = new_gs.positions[player_id]
        
        if m_type == "EGG":
            new_gs.eggs[player_id].add((curr_x, curr_y))
            points = 1
            if (curr_x in [0, self.width-1]) and (curr_y in [0, self.height-1]):
                points = 3
            new_gs.scores[player_id] += points
            
        elif m_type == "TURD":
            new_gs.turds[player_id].add((curr_x, curr_y))
            new_gs.turds_left[player_id] -= 1

        new_gs.positions[player_id] = (curr_x + dx, curr_y + dy)
        
        new_gs.p0_pos = new_gs.positions[0]
        new_gs.p1_pos = new_gs.positions[1]
        new_gs.p0_score = new_gs.scores[0]
        new_gs.p1_score = new_gs.scores[1]
        new_gs.p0_turds_left = new_gs.turds_left[0]
        new_gs.p1_turds_left = new_gs.turds_left[1]

        new_gs.turn_count += 1
        
        if new_gs.turn_count >= 80:
            new_gs.is_game_over = True
            if new_gs.scores[0] > new_gs.scores[1]:
                new_gs.winner = 0
            elif new_gs.scores[1] > new_gs.scores[0]:
                new_gs.winner = 1
            else:
                new_gs.winner = None 

        return new_gs

    def is_blocked(self, x, y, player_id):
        opp_id = 1 - player_id
        if (x, y) == self.positions[opp_id]: return True
        if (x, y) in self.eggs[opp_id]: return True
        if (x, y) in self.turds[opp_id]: return True
        for tx, ty in [(0,1), (0,-1), (1,0), (-1,0)]:
            if (x + tx, y + ty) in self.turds[opp_id]:
                return True
        return False