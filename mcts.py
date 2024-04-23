from __future__ import annotations

import multiprocessing
from typing import Tuple
import numpy as np
from game import *
from network import AlphaZeroNet, get_policy_and_value
from numba import jit, prange, float64, int64, int8
import torch
import time


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
        total_visits = np.sum(self.edges[:, 0])
        ucb_arr = c_puct * self.edges[:, 2] * np.sqrt(total_visits) / (1 + self.edges[:, 0])
        q_arr = self.edges[:, 3]
        action_idx = np.argmax(q_arr + ucb_arr)
        action = self.edges[action_idx, 4].astype(int)
        step(board, action)
        board *= -1
        return self.children[action_idx].select(board, c_puct)

    def expand(self, prior_dist: np.array, actions: np.array) -> None:
        self.edges = np.zeros((prior_dist.size, 5))
        self.edges[:, 2] = prior_dist
        self.edges[:, 4] = actions
        self.children = [Node(self, i) for i in range(prior_dist.size)]
        self.leaf = False

    def backpropagate(self, value: float) -> None:
        if self.parent is None:
            return
        self.parent.edges[self.action_idx] += np.array([1, value, 0, 0, 0])
        self.parent.edges[self.action_idx, 3] = self.parent.edges[self.action_idx, 1] / self.parent.edges[
            self.action_idx, 0]
        if self.parent is not None:
            self.parent.backpropagate(-value)

    def get_search_policy(self, tau: float) -> np.array:
        N = self.edges[:, 0]
        return N ** (1 / tau) / np.sum(N ** (1 / tau)), self.edges[:, 4].astype(int)


def mcts_search(board: np.array, root: Node, net: AlphaZeroNet, hyperparams: dict, use_model=True) -> Tuple[
    np.array, np.array]:
    if root.leaf:
        if use_model:
            policy, value = get_policy_and_value(net, board, hyperparams)
        else:
            policy, value = np.ones(7) + np.random.randn(7) * 0.05, monte_carlo_eval(board, hyperparams["rollouts"])
            policy = policy[get_valid_moves(board)]
            policy /= np.sum(policy)
        root.expand(policy, get_valid_moves(board))
    # policy, value = get_policy_and_value(net, board, hyperparams)
    # print_board(board)
    # print(value)
    for _ in range(hyperparams['iterations']):
        board_copy = np.copy(board)
        leaf = root.select(board_copy, hyperparams['c_puct'])
        winner, terminal = winner_and_terminal(board_copy)
        if not terminal:
            if use_model:
                policy, value = get_policy_and_value(net, board_copy, hyperparams)
            else:
                policy, value = np.ones(7) + np.random.randn(7) * 0.05, monte_carlo_eval(board, hyperparams["rollouts"])
                policy = policy[get_valid_moves(board_copy)]
                policy /= np.sum(policy)

            leaf.expand(policy, get_valid_moves(board_copy))
            leaf.backpropagate(-value)
        else:
            leaf.backpropagate(winner)
    return root.get_search_policy(hyperparams['tau'])


@jit(float64(int8[:, :]), nopython=True)
def monte_carlo_rollout(board: np.array) -> float:
    done = False
    board_copy = np.copy(board).astype(np.int8)
    cur_player = 1.0
    winner = 0.0
    while not done:
        valid_actions = get_valid_moves(board_copy)
        action = np.random.choice(valid_actions)
        action = np.int8(action)
        winner, done = step(board_copy, action)
        if not done:
            cur_player *= -1.0
            board_copy *= -1

    return winner * cur_player


@jit(float64(int8[:, :], int64), nopython=True, parallel=True)
def monte_carlo_eval(board: np.array, rollouts: int) -> float:
    total = np.float64(0)
    for _ in prange(rollouts):
        total += monte_carlo_rollout(board)
    return total / rollouts if rollouts > 0 else 0.0


if __name__ == '__main__':
    start_time = time.time()
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, 0, 0, 0, 0]
    ]).astype(np.int8)
    print(monte_carlo_eval(board.astype(np.int8), 0))
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")
