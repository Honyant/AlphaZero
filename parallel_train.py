from network import AlphaZeroNet, DummyAlphaZeroNet, convert_array_torch, get_policy_and_value
import torch
import numpy as np
from mcts import Node, mcts_search
from game import step, print_board, winner_and_terminal, get_valid_moves
from rich.progress import Progress
import wandb
from eval import evaluate_model
import multiprocessing
import time
mcts_hyperparams = {
    'iterations': 200,
    'c_puct': 4,
    'tau': 1,
    'device': torch.device('cuda')
    #'mps' if torch.backends.mps.is_available() else
}

training_hyperparams = {
    'lr': 0.01,
    'l2_reg': 0.01,
    'batch_size': 512,
    'num_train_iter': 4,
    'num_episodes': 800,
    'num_episodes_per_train': 1,
    'num_episodes_per_eval': 40,
    'num_eval_games': 40,
    'num_episodes_per_save': 10,
    'max_training_buffer_size': 20000,
    'num_parallel': 64,
    'device': torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
}
USE_WANDB = False

def train():
    training_buffer = []
    if USE_WANDB:
        wandb.init(project="alphazero_connect4", config=training_hyperparams)
        wandb.run.log_code(".")
        print(wandb.run.name)
        
    net = AlphaZeroNet(board_area=42, num_actions=7, input_depth=2).to(mcts_hyperparams['device'])
    #jit the model
    net = torch.jit.script(net)
    # net.load_state_dict(torch.load('model_toasty_sun_32.pth'))
    optimizer = torch.optim.Adam(net.parameters(), lr=training_hyperparams['lr'],
                                 weight_decay=training_hyperparams['l2_reg'])
    
    total_episodes = training_hyperparams['num_episodes']
    episodes_per_train = training_hyperparams['num_episodes_per_train']
    episodes_per_eval = training_hyperparams['num_episodes_per_eval']
    episodes_per_save = training_hyperparams['num_episodes_per_save']
    max_buffer_size = training_hyperparams['max_training_buffer_size']
    num_parallel = training_hyperparams['num_parallel']

    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=total_episodes)
        for episode in range(1, total_episodes + 1):
            progress.update(task, advance=1, description=f"[red]Training... (Episode {episode}/{total_episodes})")
            training_buffer.extend(run_episodes(num_parallel, net, mcts_hyperparams))
            if len(training_buffer) > max_buffer_size:
                training_buffer = training_buffer[-max_buffer_size:]
            if episode % episodes_per_train == 0 or episode == total_episodes:
                train_network(net, optimizer, training_buffer, training_hyperparams, episode*training_hyperparams['num_parallel'])
            if episode % episodes_per_eval == 0 or episode == total_episodes:
                win_rate = evaluate_model(net, training_hyperparams['num_eval_games'])
                if USE_WANDB:
                    wandb.log({"Win Rate": win_rate}, step=episode*training_hyperparams['num_parallel'])
            # Save the model at specific intervals
            if episode % episodes_per_save == 0 or episode == total_episodes:
                torch.save(net.state_dict(), f'model_{wandb.run.name.replace("-", "_") if wandb.run else 0}.pth')

    torch.save(net.state_dict(), f'model_{wandb.run.name.replace("-", "_") if wandb.run else 0}.pth')
    if USE_WANDB:
        wandb.finish()


def train_network(network, optimizer, training_buffer, hyperparams: dict, episode):
    network.train()
    network.to(hyperparams['device'])
    for _ in range(hyperparams['num_train_iter']):
        indices = np.random.choice(len(training_buffer), hyperparams['batch_size'])
        batch = [training_buffer[i] for i in indices]  #batch should be a list of tuples
        states, search_policies, values = [np.array(x) for x in zip(*batch)]
        # states = torch.Tensor(states).flatten(start_dim=1).to(hyperparams['device'])
        states = convert_array_torch(torch.Tensor(states).to(hyperparams['device']).unsqueeze(1))

        search_policies = torch.Tensor(search_policies).to(hyperparams['device'])
        values_gt = torch.Tensor(values).to(hyperparams['device'])

        optimizer.zero_grad()
        values_gt = values_gt.unsqueeze(1)
        policy, value_pred = network(states)
        policy_loss = -torch.mean(torch.sum(search_policies * torch.log(policy), dim=1))
        value_loss = torch.nn.MSELoss()(value_pred, values_gt)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        # Print the losses using rich
        # print( f"[blue]Policy Loss:[/blue] {policy_loss.item():.4f}, [blue]Value Loss:[/blue] {value_loss.item():.4f}, [blue]Total Loss:[/blue] {loss.item():.4f}")

        # Log the losses using wandb
        if USE_WANDB:
            wandb.log({
                "Policy Loss": policy_loss.item(),
                "Value Loss": value_loss.item(),
                "Total Loss": loss.item()
            }, step=episode)
    # network.to(torch.device('cpu'))
    network.eval()
    
def worker(process_id, hyperparams, out_board, result_policy, result_value, batch_queue, barrier):
    np.random.seed(process_id)
    states = []
    search_policies = []
    board = np.zeros((6, 7)).astype(np.int8)
    root = Node(None, None)
    cur_player = 1
    personal_done = False
    count = 0
    while not personal_done:
        states.append(np.copy(board))
        for n in range(hyperparams['iterations']):
            count +=1
            board_copy = np.copy(board)
            leaf = root.select(board_copy, hyperparams['c_puct'])
            winner, terminal = winner_and_terminal(board_copy)
            out_board[process_id] = board_copy
            barrier.wait()
            barrier.wait()
            mcts_policy = result_policy[process_id]
            mcts_value = result_value[process_id]
            # barrier.wait()
            if n%100 == 0 and (process_id == 0 or process_id == 1):
                print(n)
                print_board(board_copy)
            if not terminal:
                mcts_policy = mcts_policy[get_valid_moves(board_copy)]
                if np.sum(mcts_policy) == 0:
                    mcts_policy = np.ones(mcts_policy.shape[0])
                mcts_policy /= np.sum(mcts_policy)
                leaf.expand(mcts_policy, get_valid_moves(board_copy))
                leaf.backpropagate(-mcts_value)
            else:
                leaf.backpropagate(winner)
        
        
        policy, actions = root.get_search_policy(hyperparams['tau'])
        action_idx = np.random.choice(len(actions), p=policy)
        final_value, personal_done = step(board, actions[action_idx])
        root = root.children[action_idx]
        complete_policy = np.zeros(7)
        complete_policy[actions] = policy
        search_policies.append(complete_policy)
        board *= -1
        if not personal_done:
            cur_player *= -1
            
    values = np.ones(len(states))
    if cur_player == 1:
        values[1::2] = -1
    else:
        values[::2] = -1
    values = list(values * final_value)
    
    while count < 42 * hyperparams['iterations']:
        count +=1
        barrier.wait()
        barrier.wait()
    
    barrier.wait()
    batch_queue.put((states, search_policies, values))

def run_episodes(num_parallel, net, hyperparams: dict):
    net.to(hyperparams['device'])
    batch_queue = multiprocessing.Queue()
    barrier = multiprocessing.Barrier(num_parallel + 1)
    states = []
    out_board = np.zeros((num_parallel, 6, 7)).astype(np.int8)
    result_policy = np.zeros((num_parallel, 7))
    result_value = np.zeros(num_parallel)
    search_policies = []
    processes = []
    for i in range(num_parallel):
        process = multiprocessing.Process(target=worker, args=(i, hyperparams, out_board, result_policy, result_value, batch_queue, barrier))
        processes.append(process)
        process.start()
    count = 0
    while count < 42 * hyperparams['iterations']:
        count += 1
        barrier.wait()
        mcts_policy, mcts_value = net(convert_array_torch(torch.Tensor(out_board).unsqueeze(1).to(hyperparams['device'])))
        result_policy = mcts_policy.detach().cpu().numpy()
        result_value = mcts_value.detach().cpu().numpy().flatten()
        barrier.wait()
        # barrier.wait()
        
    all_states = []
    all_search_policies = []
    all_values = []
    barrier.wait()
    
    for i in range(num_parallel):
        states, search_policies, values = batch_queue.get()
        all_states.extend(states)
        all_search_policies.extend(search_policies)
        all_values.extend(values)
    return (zip(all_states, all_search_policies, all_values))

if __name__ == "__main__":
    train()