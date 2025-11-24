"""
Monte Carlo Tree Search (MCTS) with UCT
Inspired by AlphaZero but simplified for time constraints
"""
import math
import random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from game.board import Board
from game.enums import Direction, MoveType


class MCTSNode:
    """
    Node in the Monte Carlo search tree.
    """
    
    def __init__(self, board: Board, move: Optional[Tuple[Direction, MoveType]] = None,
                 parent: Optional['MCTSNode'] = None):
        self.board = board
        self.move = move  # Move that led to this node
        self.parent = parent
        
        # MCTS statistics
        self.visits = 0
        self.total_value = 0.0
        self.children: List['MCTSNode'] = []
        self.untried_moves: List[Tuple[Direction, MoveType]] = []
        
        # Initialize untried moves
        if not board.is_game_over():
            self.untried_moves = board.get_valid_moves()
    
    def is_fully_expanded(self) -> bool:
        """Check if all children have been created."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.board.is_game_over()
    
    def uct_value(self, exploration_constant: float = 1.41) -> float:
        """
        Calculate UCT (Upper Confidence Bound for Trees) value.
        Balance exploitation (high value) vs exploration (low visits).
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have max priority
        
        # UCT formula: Q/N + c * sqrt(ln(N_parent) / N)
        exploitation = self.total_value / self.visits
        
        if self.parent and self.parent.visits > 0:
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
        else:
            exploration = 0
        
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.41) -> 'MCTSNode':
        """Select child with highest UCT value."""
        return max(self.children, key=lambda c: c.uct_value(exploration_constant))
    
    def most_visited_child(self) -> Optional['MCTSNode']:
        """Select most visited child (for final move selection)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)


class MCTS:
    """
    Monte Carlo Tree Search with UCT selection.
    """
    
    def __init__(self, exploration_constant: float = 1.41):
        self.exploration_constant = exploration_constant
        self.node_cache: Dict[int, MCTSNode] = {}
    
    def search(self, root_board: Board, num_simulations: int,
               evaluation_func=None) -> Tuple[Direction, MoveType]:
        """
        Run MCTS for specified number of simulations.
        
        Args:
            root_board: Current board state
            num_simulations: Number of MCTS iterations
            evaluation_func: Optional function to evaluate leaf nodes (if None, use rollout)
        
        Returns:
            Best move (Direction, MoveType)
        """
        root = MCTSNode(root_board)
        
        for _ in range(num_simulations):
            # 1. Selection: traverse tree using UCT
            node = self._select(root)
            
            # 2. Expansion: add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            
            # 3. Simulation: evaluate position
            if evaluation_func:
                value = evaluation_func(node.board)
            else:
                value = self._simulate(node.board)
            
            # 4. Backpropagation: update statistics
            self._backpropagate(node, value)
        
        # Return move of most visited child
        best_child = root.most_visited_child()
        if best_child and best_child.move:
            return best_child.move
        
        # Fallback: return first valid move
        valid_moves = root_board.get_valid_moves()
        return valid_moves[0] if valid_moves else (Direction.UP, MoveType.PLAIN)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCT until we find unexpanded node.
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            
            # Select best child using UCT
            node = node.best_child(self.exploration_constant)
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: add one new child to the tree.
        """
        if not node.untried_moves:
            return node
        
        # Pick random untried move
        move = node.untried_moves.pop()
        
        # Create new board state
        next_board = node.board.forecast_move(move[0], move[1], check_ok=False)
        if next_board is None:
            # Invalid move, try another
            if node.untried_moves:
                return self._expand(node)
            return node
        
        # Create child node
        child = MCTSNode(next_board, move, parent=node)
        node.children.append(child)
        
        return child
    
    def _simulate(self, board: Board, max_depth: int = 20) -> float:
        """
        Simulation phase: play out game randomly (rollout policy).
        Returns value from perspective of current player.
        """
        current_board = board
        depth = 0
        
        while not current_board.is_game_over() and depth < max_depth:
            moves = current_board.get_valid_moves()
            if not moves:
                break
            
            # Simple rollout policy: prefer egg moves, then random
            egg_moves = [m for m in moves if m[1] == MoveType.EGG]
            if egg_moves and random.random() < 0.7:
                move = random.choice(egg_moves)
            else:
                move = random.choice(moves)
            
            next_board = current_board.forecast_move(move[0], move[1], check_ok=False)
            if next_board is None:
                break
            
            current_board = next_board
            depth += 1
        
        # Evaluate final position
        return self._evaluate_position(current_board, board)
    
    def _evaluate_position(self, final_board: Board, initial_board: Board) -> float:
        """
        Evaluate position from initial player's perspective.
        Returns value in [-1, 1].
        """
        # Get egg counts
        my_eggs = final_board.chicken_player.get_eggs_laid()
        enemy_eggs = final_board.chicken_enemy.get_eggs_laid()
        
        # Determine if we're the same player
        # (board perspective might have flipped during rollout)
        same_player = (final_board.chicken_player.get_spawn() == 
                      initial_board.chicken_player.get_spawn())
        
        if not same_player:
            # Perspective flipped
            my_eggs, enemy_eggs = enemy_eggs, my_eggs
        
        # Normalize to [-1, 1]
        # Assume max eggs is ~20
        diff = my_eggs - enemy_eggs
        value = diff / 20.0
        
        # Clamp to [-1, 1]
        value = max(-1.0, min(1.0, value))
        
        return value
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagation phase: update node statistics up the tree.
        """
        while node is not None:
            node.visits += 1
            node.total_value += value
            
            # Flip value for opponent (minimax-style)
            value = -value
            
            node = node.parent
