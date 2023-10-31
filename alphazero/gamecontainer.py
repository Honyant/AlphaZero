from alphazero.game import Game
import numpy as np
from alphazero.mcts import Node

class GameContainer:

    def __init__(self):
        self.state_history = []
        self.child_visits = []
        self.action_history = []
        self.value_history = []
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
    
    def sample_move(self, action_probs, node, tau, c_puct):
        #sum of all visit counts of children:

        def modify_counts(counts):
            return counts ** (1 / tau)

        total_visits = sum(modify_counts(child.visits) for child in node.children)
        for action, prob in enumerate(action_probs):
            if action in self.get_moves():
                node.add_child(self.game.board, action)
                node.children[-1].probability = prob

        def puct(node):
            return node.mean_value + c_puct * node.probability * np.sqrt(total_visits) / (1 + node.visits)
        
        #get action with puct values:
        puct_values = [puct(child) for child in node.children]
        puct_values = np.array(puct_values)
        puct_values = puct_values / sum(puct_values)

        #add dirichlet noise:
        noise = np.random.dirichlet([0.03] * len(puct_values))
        puct_values = 0.75 * puct_values + 0.25 * noise

        #get highest value:
        action = np.argmax(puct_values)
        #update state:
        self.make_move(action)
        self.state_history.append(self.game.board)
        self.action_history.append(action)
        self.child_visits.append([child.visits for child in node.children])
        self.value_history.append(node.mean_value)
        return action

    def remove_illegal_moves(self,action_probs, possible_moves):
        probs = np.zeros(7)
        for move in possible_moves:
            probs[move] = 1
        probs = probs * action_probs
        probs = probs / sum(probs)
        return probs

    def find_leaf(self, root: Node, tau, c_puct, predictions):
        node = root
        while node.children:
            predictions = self.remove_illegal_moves(predictions, np.arange(7))
            action, node = self.sample_move(node.probability, node, tau=tau, c_puct=c_puct)
            self.make_move(action)
        return node
    

    def backpropagate(self, node: Node, value):
        while node:
            node.update(value)
            node = node.parent

    def evaluate(self):
        winner = self.game.get_winner()
        return 0 if winner == None else winner
    
    def run_mcts(self, root: Node, num_simulations, tau, c_puct):
        for _ in range(num_simulations):
            leaf = self.find_leaf(root, tau, c_puct)
            value = self.evaluate(leaf)
            self.backpropagate(leaf, value)