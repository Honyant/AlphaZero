from alphazero.game import Game
from alphazero.gamecontainer import GameContainer
from alphazero.model import AlphaZeroNet
import torch
import numpy as np
import time
#test the game functions:
def testGameplay():
    net = AlphaZeroNet(board_area=6*7, num_actions=7, input_depth=8)

    gc = GameContainer()
    
    while not gc.game.terminal():

        print(gc.game.board)
        print("Player", gc.game.player, "to play.")
        print("Possible moves:", gc.get_moves())
        action_probs, value = net(torch.unsqueeze(torch.from_numpy(gc.get_state()*gc.game.get_player()).float(), 0))
        action_probs = action_probs.squeeze().detach().numpy()
        value = value.squeeze().detach().numpy()
        possible_moves = gc.get_moves()

        #remove illegal moves by setting their probability to 0
        probs = remove_illegal_moves(action_probs, possible_moves)
        

        move = np.random.choice(np.arange(7), p=probs)
        gc.make_move(move)
        print("Action probabilities:", action_probs)
        print("Value:", value)
    print(gc.game.board)
    if gc.game.get_winner() == 0:
        print("Draw!")
    else:
        print("Player", gc.game.get_winner(), "wins!")

def remove_illegal_moves(action_probs, possible_moves):
    probs = np.zeros(7)
    for move in possible_moves:
        probs[move] = 1
    probs = probs * action_probs
    probs = probs / sum(probs)
    return probs

for i in range(100):
    testGameplay()
    time.sleep(1)