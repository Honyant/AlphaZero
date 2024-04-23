import numpy as np
from mcts import Node, mcts_search
from game import step, print_board, get_valid_moves
import torch
from network import AlphaZeroNet, get_policy_and_value

mcts_hyperparams = {
        'iterations': 1000,
        'c_puct': 1.0,
        'tau': 1,
        'device': torch.device('cpu'),
        'rollouts': 1000,
        #'mps' if torch.backends.mps.is_available() else 
}

def human_play(network, starting_player, hyperparams : dict):
    board = np.zeros((6,7)).astype(np.int8)
    done = False
    cur_player = starting_player
    
    while not done:
        if cur_player == 1:
            # print(board.astype(int))
            print_board(board)
            # Human player's turn
            while True:
                player_input = input("Enter your move (0-6): ")
                if player_input == "exit":
                    return
                if not player_input.isdigit():
                    print("Invalid input. Try again.")
                    continue
                action = int(player_input)
                if 0 <= action <= 6 and board[0, action] == 0:
                    break
                print("Invalid move. Try again.")
        else:
            # AI player's turn
            root = Node(None, None)
            # print(board)
            # policy, value = get_policy_and_value(network, board, hyperparams)
            # actions = get_valid_moves(board)
            policy, actions = mcts_search(board, root, network, hyperparams, use_model=False)
            print(policy)
            action_idx = np.argmax(policy)
            action = actions[action_idx]
        final_value, done = step(board, action)
        if not done:
            board *= -1
            cur_player *= -1
    print(cur_player, final_value, done)
    
    if cur_player == -1:
        board *= -1
    print_board(board)
    if final_value == 0:
        print("It's a draw!")
    elif cur_player == 1:
        print("Player 1 wins!")
    elif cur_player == -1:
        print("Player 2 wins!")

if __name__ == "__main__":
    # Load the trained model
    net = AlphaZeroNet(board_area=42, num_actions=7, input_depth=2).to(mcts_hyperparams['device'])
    
    net.load_state_dict(torch.load('model_drawn_sunset_42.pth'))
    net.eval()
    # Start the game
    # starting_player = 1 if np.random.rand() < 0.5 else -1
    starting_player = 1
    if starting_player == 1:
        print("You are player 1.")
    else:
        print("You are player 2.")
    human_play(net, starting_player, mcts_hyperparams)
