from alphazero.model import AlphaZeroNet
from alphazero.game import Game
from alphazero.gamecontainer import GameContainer
import torch

def test_print():
    net = AlphaZeroNet(board_area=6*7, num_actions=7, input_depth=8)
    print(net)
def test_model():
    net = AlphaZeroNet(board_area=6*7, num_actions=7, input_depth=8)
    out = net(torch.randn(1, 8, 6, 7))
    print("Action probabilities:", out[0].squeeze().detach().numpy())
    print("Value:", out[1].squeeze().detach().numpy())

test_model()