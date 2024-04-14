from typing import Tuple
import numpy as np

def step(board: np.array, move: int) -> Tuple[int, bool]: # reward, done
    for i in range(5, -1, -1):
        if board[i][move] == 0:
            board[i][move] = 1 # you are always the player
            break
    winner, terminal = winner_and_terminal(board)
    return winner, terminal
    
def get_valid_moves(board: np.array) -> np.array:
    return np.where(board[0] == 0)[0]

def get_winner(board: np.array) -> int:
    rows, cols = board.shape
    # Check horizontal
    for row in range(rows):
        for col in range(cols - 3):
            line_sum = sum(board[row][col:col+4])
            if abs(line_sum) == 4:
                return board[row][col]

    # Check vertical
    for col in range(cols):
        for row in range(rows - 3):
            line_sum = board[row][col] + board[row+1][col] + board[row+2][col] + board[row+3][col]
            if abs(line_sum) == 4:
                return board[row][col]

    # Check diagonals
    for row in range(rows - 3):
        for col in range(cols - 3):
            # Positive slope
            pos_slope_sum = board[row][col] + board[row+1][col+1] + board[row+2][col+2] + board[row+3][col+3]
            # Negative slope
            neg_slope_sum = board[row+3][col] + board[row+2][col+1] + board[row+1][col+2] + board[row][col+3]

            if abs(pos_slope_sum) == 4:
                return board[row][col]
            elif abs(neg_slope_sum) == 4:
                return board[row+3][col]

    return None

def winner_and_terminal(board: np.array) -> Tuple[int, bool]:
    full_board = len(get_valid_moves(board)) == 0
    winner = get_winner(board)
    return 0 if winner == None else 1, winner != None or full_board