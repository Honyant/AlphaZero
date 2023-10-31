from alphazero.game import Game
from alphazero.model import AlphaZeroNet
import torch
import numpy as np
from alphazero.utils import remove_illegal_moves
from alphazero.gamecontainer import GameContainer

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#test the game functions:
net = torch.jit.script(AlphaZeroNet(board_area=6*7, num_actions=7, input_depth=8))
net.to(device)

#play game with input:
def play_game(game: Game):
    while not game.terminal():
        print(game.board)
        print("Player", game.player, "to play.")
        move = int(input("Enter a move: "))
        game.apply(move)
    print(game.board)
    if game.get_winner() == 0:
        print("Draw!")
    else:
        print("Player", game.get_winner(), "wins!")


def play_against_ai(game: GameContainer):
    while not game.game.terminal():
        print(game.game.board)
        print("Player", game.game.player, "to play.")
        if game.game.player == 1:
            move = int(input("Enter a move: "))
            game.make_move(move)
        else:
            model_input = torch.unsqueeze(torch.from_numpy(game.get_state()*game.game.get_player()).float(), 0)
            action_probs, value = net(model_input.to(device))
            action_probs = action_probs.squeeze().detach().cpu().numpy()
            value = value.squeeze().detach().cpu().numpy()
            possible_moves = game.get_moves()
            probs = remove_illegal_moves(action_probs, possible_moves)
            move = np.random.choice(np.arange(7), p=probs)
            game.make_move(move)
    print(game.game.board)
    if game.game.get_winner() == 0:
        print("Draw!")
    else:
        print("Player", game.game.get_winner(), "wins!")

#play_game(Game())
play_against_ai(GameContainer())