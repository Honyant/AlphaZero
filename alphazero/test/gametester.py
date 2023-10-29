import numpy as np
import pytest
from alphazero.game import Game

def test_initialization():
    game = Game()
    assert game.get_player() == 1
    assert game.get_winner() is None
    assert np.array_equal(game.board, np.zeros((6, 7), dtype=np.int8))
    assert not game.terminal()

def test_set_board():
    game = Game()
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int8)
    game.set_board(board)
    assert game.get_winner() is None
    assert game.get_player() == 1

def test_get_moves():
    game = Game()
    assert set(game.get_moves()) == {0, 1, 2, 3, 4, 5, 6}
    game.apply(0)
    assert set(game.get_moves()) == {0, 1, 2, 3, 4, 5, 6}

def test_terminal():
    game = Game()
    board = np.ones((6, 7), dtype=np.int8)
    game.set_board(board)
    assert game.terminal()
    assert game.get_winner() == 0

def test_apply_and_winner():
    game = Game()
    for i in range(4):
        game.apply(i)
    assert game.get_winner() is None
    game.apply(4)
    assert game.get_winner() == 1

    game = Game()
    for _ in range(3):
        game.apply(0)
        game.apply(1)
    game.apply(0)
    assert game.get_winner() == 1

def test_diagonal_winner():
    game = Game()
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [1, 1, -1, 0, 0, 0, 0],
        [1, 1, 1, -1, 0, 0, 0],
    ], dtype=np.int8)
    game.set_board(board)
    assert game.get_winner() == -1

if __name__ == "__main__":
    pytest.main([__file__])
