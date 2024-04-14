import numpy as np
from mcts import Node, mcts_search
from game import step
import torch
from network import AlphaZeroNet

mcts_hyperparams = {
        'iterations': 100,
        'c_puct': 1.0,
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
            print(board.astype(int))
            # Human player's turn
            while True:
                action = int(input("Enter your move (0-6): "))
                if 0 <= action <= 6 and board[0, action] == 0:
                    break
                print("Invalid move. Try again.")
        else:
            # AI player's turn
            root = Node(None, None)
            policy, actions = mcts_search(board, root, network, hyperparams)
            print(policy)
            action_idx = np.random.choice(len(actions), p=policy)
            action = actions[action_idx]
        
        final_value, done = step(board, action)
        if done:
            break
        cur_player *= -1
        board *= -1
    print(board.astype(int))
    if cur_player == 1:
        print("Player 1 wins!")
    elif cur_player == -1:
        print("Player 2 wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    # Load the trained model
    net = AlphaZeroNet(board_area=42, num_actions=7, input_depth=1).to(mcts_hyperparams['device'])
    net.load_state_dict(torch.load('model.pth'))
    net.eval()
    
    # Start the game
    starting_player = 1 if np.random.rand() < 0.5 else -1
    if starting_player == 1:
        print("You are player 1.")
    else:
        print("You are player 2.")
    human_play(net, starting_player, mcts_hyperparams)