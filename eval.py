import numpy as np
from mcts import Node, mcts_search
from game import step, print_board
import torch
from network import AlphaZeroNet

mcts_hyperparams = {
        'iterations': 200,
        'c_puct': 4.0,
        'tau': 1,
        'device': torch.device('cpu')
        #'mps' if torch.backends.mps.is_available() else 
}

def human_play(network, starting_player, hyperparams : dict):
    board = np.zeros((6,7))
    done = False
    cur_player = starting_player
    
    while not done:
        if cur_player == 1:
            # MCTS player's turn
            root = Node(None, None)
            policy, actions = mcts_search(board, root, network, hyperparams, use_model=False)
            action_idx = np.random.choice(len(actions), p=policy)
            # action_idx = np.argmax(policy)
            action = actions[action_idx]
        else:
            # AI player's turn
            root = Node(None, None)
            policy, actions = mcts_search(board, root, network, hyperparams)
            action_idx = np.random.choice(len(actions), p=policy)
            # action_idx = np.argmax(policy)
            action = actions[action_idx]
        final_value, done = step(board, action)
        if not done:
            board *= -1
            cur_player *= -1
    if final_value == 0 and done:
        return 0
    return cur_player

def game(net): # returns if the ai won or not
    # Start the game
    starting_player = 1 if np.random.rand() < 0.5 else -1
    result = human_play(net, starting_player, mcts_hyperparams)
    return 0.5 if result == 0 else int(-starting_player == result)

def evaluate_model(model, games):
    model.eval()
    stats = []
    for _ in range(games):
        stats.append(game(model))
        print(stats[-1])
    print(mcts_hyperparams)
    model.train()
    return sum(stats) / games

if __name__ == "__main__":
    net = AlphaZeroNet(board_area=42, num_actions=7, input_depth=2).to(mcts_hyperparams['device'])
    net.load_state_dict(torch.load('model_hondhzd7.pth'))
    print(f'Win rate: {evaluate_model(net, 100)}')