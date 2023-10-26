#connect4 game
import numpy as np
class Game:
    def __init__(self):#player is 1 or -1
        self.board = np.zeros((6, 7), dtype=np.int8)
        self.player = 1
        self.winner = None # 1 or -1 for player, 0 for draw, None for not terminal
    
    def set_board(self, board: np.ndarray):
        self.board = board.copy()
        self.check_winner()

    def get_moves(self):
        return np.where(self.board[0] == 0)[0]
    
    def get_player(self):
        return self.player
    
    def terminal(self):
        return self.winner != None or len(self.get_moves()) == 0
    
    def get_winner(self):
        return self.winner
    
    def clone(self):
        g = Game()
        g.board = self.board.copy()
        g.player = self.player
        g.winner = self.winner
        return g
    
    def apply(self, move):
        for i in range(5, -1, -1):
            if self.board[i][move] == 0:
                self.board[i][move] = self.player
                break
        self.player *= -1
        self.check_winner()

    def check_winner(self):
        # Check horizontal locations for winner
        for row in range(6):
            for col in range(4):
                if abs(sum(self.board[row, col:col+4])) == 4:
                    self.winner = self.board[row, col]
                    return

        # Check vertical locations for winner
        for col in range(7):
            for row in range(3):
                if abs(sum(self.board[row:row+4, col])) == 4:
                    self.winner = self.board[row, col]
                    return

        # Check positive sloped diagonals
        for row in range(3):
            for col in range(4):
                if abs(self.board[row, col] + self.board[row+1, col+1] + self.board[row+2, col+2] + self.board[row+3, col+3]) == 4:
                    self.winner = self.board[row, col]
                    return

        # Check negative sloped diagonals
        for row in range(3, 6):
            for col in range(4):
                if abs(self.board[row, col] + self.board[row-1, col+1] + self.board[row-2, col+2] + self.board[row-3, col+3]) == 4:
                    self.winner = self.board[row, col]
                    return

        if len(self.get_moves()) == 0:
            self.winner = 0
            return
        
        return