import numpy as np
from mcts import Node, mcts_search
from game import step, print_board, get_valid_moves, winner_and_terminal
import torch
from network import AlphaZeroNet, get_policy_and_value, convert_array_torch
import multiprocessing
import time
mcts_hyperparams = {
        'iterations': 200,
        'c_puct': 4.0,
        'tau': 1,
        'device': torch.device('cuda')
}

def worker(process_id, hyperparams, batch_queue, barrier):
    np.random.seed(process_id)
    board = np.zeros((6, 7)).astype(np.int8)
    root = Node(None, None)
    cur_player = 1 if process_id % 2 == 0 else -1
    personal_done = False
    count = 0
    while not personal_done:
        root = Node(None, None)
        for n in range(hyperparams['iterations']):
            count +=1
            board_copy = np.copy(board)
            leaf = root.select(board_copy, hyperparams['c_puct'])
            winner, terminal = winner_and_terminal(board_copy)
            batch_queue.put((process_id, board_copy))
            barrier.wait()
            barrier.wait()
            mcts_policy, mcts_value = batch_queue.get()
            barrier.wait()
            mcts_policy = mcts_policy[get_valid_moves(board_copy)]
            mcts_policy /= np.sum(mcts_policy)
            if cur_player == 1:
                mcts_policy, mcts_value = np.ones(7) + np.random.randn(7) * 0.05, 0
                mcts_policy = mcts_policy[get_valid_moves(board_copy)]
                mcts_policy /= np.sum(mcts_policy)
            # if process_id == 0 and n%100 == 0:
            #     print(mcts_policy, mcts_value, cur_player)     
            if not terminal:
                leaf.expand(mcts_policy, get_valid_moves(board_copy))
                leaf.backpropagate(-mcts_value)
            else:
                leaf.backpropagate(winner)
        policy, actions = root.get_search_policy(hyperparams['tau'])
        if process_id == 0:

            print(policy, actions, cur_player)
        action_idx = np.argmax(policy)
        final_value, personal_done = step(board, actions[action_idx])
        board *= -1
        if not personal_done:
            cur_player *= -1
    print(cur_player, final_value, personal_done)
    
    while count < 42 * hyperparams['iterations']:
        count += 1
        batch_queue.put((process_id, board))
        barrier.wait()
        barrier.wait()
        mcts_policy, mcts_value = batch_queue.get()
        barrier.wait()
    
    barrier.wait()
    print(cur_player, final_value, personal_done)
    batch_queue.put(0.5 if final_value == 0 else int(cur_player==-1))

def run_games(num_parallel, net, hyperparams: dict):
    net.to(hyperparams['device'])
    batch_queue = multiprocessing.Queue()
    barrier = multiprocessing.Barrier(num_parallel + 1)
    processes = []
    for i in range(num_parallel):
        process = multiprocessing.Process(target=worker, args=(i, hyperparams, batch_queue, barrier))
        processes.append(process)
        process.start()
    count = 0
    while count < 42 * hyperparams['iterations']:
        count += 1
        batch_boards = []
        batch_ids = []
        barrier.wait()
        for i in range(num_parallel):
            process_id, board = batch_queue.get()
            batch_boards.append(board)
            batch_ids.append(process_id)
        barrier.wait()
        batch_boards = np.array(batch_boards)
        mcts_policy, mcts_value = net(convert_array_torch(torch.Tensor(batch_boards).unsqueeze(1).to(hyperparams['device'])))
        mcts_policy = mcts_policy.detach().cpu().numpy()
        mcts_value = mcts_value.detach().cpu().numpy().flatten()
        
        for i, process_id in enumerate(batch_ids):
            batch_queue.put((mcts_policy[i], mcts_value[i]))
        barrier.wait()
        
    barrier.wait()
    results = []
    for i in range(num_parallel):
        results.append(batch_queue.get())
    print(results)
    return sum(results)/num_parallel

def evaluate_model(model, games, print_game = False):
    model.eval()
    # model.to(mcts_hyperparams['device'])
    win_rate = run_games(games, model, mcts_hyperparams)
    print(f'Win rate: {win_rate}')
    model.train()
    # model.to(torch.device('cuda'))
    return win_rate

if __name__ == "__main__":
    net = AlphaZeroNet(board_area=42, num_actions=7, input_depth=2).to(mcts_hyperparams['device'])
    net.load_state_dict(torch.load('model_drawn_sunset_42.pth'))
    net = torch.jit.script(net)
    print(f'Win rate: {evaluate_model(net, 32, print_game=True)}')