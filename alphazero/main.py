from alphazero.game import Game
import numpy as np
from alphazero.mcts import Node
from alphazero.gamecontainer import GameContainer
from alphazero.model import AlphaZeroNet
import torch
from alphazero.dataset import AlphaZeroDataset
from alphazero.train import train
from alphazero.utils import remove_illegal_moves
# Set up hyperparameters
tau = 1
c_puct = 4
num_simulations = 800
num_games = 100
num_epochs = 10
num_batches = 100
num_epochs_per_iteration = 10
num_games_per_iteration = 100


# Set up the neural network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.jit.script(AlphaZeroNet(board_area=6*7, num_actions=7, input_depth=8))
net.to(device)
net(torch.from_numpy(np.zeros((1, 8, 6, 7))).float().to(device))

# Set up the game container
gc = GameContainer()

def main():
    # Train the neural network
    for i in range(num_batches):
        print("Batch", i)
        # self-play
        for j in range(num_games_per_iteration):
            print("Game", j)
            gc = GameContainer()
            while not gc.game.terminal():
                model_input = torch.unsqueeze(torch.from_numpy(gc.get_state()*gc.game.get_player()).float(), 0)
                action_probs, value = net(model_input.to(device))
                action_probs = action_probs.squeeze().detach().cpu().numpy()
                value = value.squeeze().detach().cpu().numpy()
                gc.child_visits.append(action_probs)
                gc.value_history.append(value)
                #random:
                #action_probs = np.random.rand(7)
                #value = np.random.rand(1)
                possible_moves = gc.get_moves()
                #remove illegal moves by setting their probability to 0
                probs = remove_illegal_moves(action_probs, possible_moves)
                move = np.random.choice(np.arange(7), p=probs)
                gc.make_move(move)
            if gc.game.get_winner() == 0:
                print("Draw!")
            else:
                print("Player", gc.game.get_winner(), "wins!")
        # train the neural network
        dataset = AlphaZeroDataset(gc)
        dataset.init_from_replay()
        train(net, dataset, num_epochs_per_iteration, batch_size=64)
        # save the neural network
        torch.save(net.state_dict(), "model.pt")