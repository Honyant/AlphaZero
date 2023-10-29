from alphazero.game import Game
import numpy as np
class GameContainer:

    def __init__(self):
        self.state_history = []
        self.child_visits = []
        self.action_history = []
        self.game = Game()
        #store the game state history with 8 timesteps (we will feed this to the neural network)
        self.state = np.zeros((8,6,7), dtype=np.int8)
    
    def get_state(self):
        return self.state
    
    
    def make_move(self, move):
        self.game.apply(move)
        self.update_state()

    def update_state(self):
        self.state_history.append(self.game.board)
        #shape
        self.state = np.vstack((self.game.board[np.newaxis, :], self.state[:-1]))

    def get_moves(self):
        return self.game.get_moves()
    
    