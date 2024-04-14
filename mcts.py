from __future__ import annotations
from typing import Tuple
import numpy as np
from game import *
from network import AlphaZeroNet
import torch

 
class Node:
    def __init__(self, parent: Node, action_idx: int) -> None:
        self.children = []
        self.parent = parent
        self.action_idx = action_idx
        self.edges = None
        self.leaf = True
    
    def select(self, board: np.array, c_puct: float) -> Node:
        # N W P Q action (5)
        # 0 1 2 3 4
        if self.leaf:
            return self
        total_visits = np.sum(self.edges[:,0])
        ucb_arr = c_puct * self.edges[:,2] * np.sqrt(total_visits) / (1 + self.edges[:,0])
        q_arr = self.edges[:,3]
        action_idx = np.argmax(q_arr + ucb_arr)
        action = self.edges[action_idx, 4].astype(int)
        board *= -1
        step(board, action)
        # print(np.sum(np.abs(board)))
        return self.children[action_idx].select(board, c_puct)
        
    def expand(self, prior_dist: np.array, actions: np.array) -> None:
        self.edges = np.zeros((prior_dist.size, 5))
        self.edges[:, 2] = prior_dist
        self.edges[:, 4] = actions
        self.children = [Node(self, i) for i in range(prior_dist.size)]
        self.leaf = False
        
    def backpropagate(self, value : float) -> None:
        if self.parent is None:
            return
        self.parent.edges[self.action_idx] += np.array([1, value, 0, 0, 0])
        self.parent.edges[self.action_idx, 3] = self.parent.edges[self.action_idx, 1] / self.parent.edges[self.action_idx, 0]
        if self.parent is not None:
            self.parent.backpropagate(-value)
    
    def get_search_policy(self, tau: float) -> np.array:
        N = self.edges[:, 0]
        return N**(1/tau) / np.sum(N**(1/tau)), self.edges[:, 4].astype(int)
    
def mcts_search(board: np.array, root: Node, net: AlphaZeroNet, hyperparams: dict) -> Tuple[np.array, np.array]:
    for _ in range(hyperparams['iterations']):
        board_copy = np.copy(board)
        leaf = root.select(board_copy, hyperparams['c_puct'])
        winner, terminal = winner_and_terminal(board_copy)
        if not terminal:
            policy, value = net(torch.Tensor(board_copy).to(hyperparams['device']).unsqueeze(0).unsqueeze(0))
            policy, value = policy.detach().cpu().numpy().flatten(), value.detach().cpu().numpy().flatten().item() # go inside network
            #mask policy
            policy = (policy * (1-np.abs(board_copy[0])))
            policy = policy[policy != 0] / np.sum(policy)
            leaf.expand(policy.flatten(), get_valid_moves(board_copy))
            leaf.backpropagate(value)
        else:
            leaf.backpropagate(winner)
    return root.get_search_policy(hyperparams['tau'])
