from network import AlphaZeroNet, DummyAlphaZeroNet, convert_array_torch
import torch
import numpy as np
from mcts import Node, mcts_search
from game import step, print_board
from rich.progress import Progress
import wandb

mcts_hyperparams = {
    'iterations': 25,
    'c_puct': 4.0,
    'tau': 1,
    'device': torch.device('cpu')
    #'mps' if torch.backends.mps.is_available() else
}

training_hyperparams = {
    'lr': 0.01,
    'l2_reg': 0.01,
    'batch_size': 1024,
    'num_train_iter': 1,
    'num_episodes': 10000,
    'num_episodes_per_train': 4,
    'device': torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
}
USE_WANDB = True

def train():
    training_buffer = []
    if USE_WANDB:
        wandb.init(project="alphazero_connect4", config=training_hyperparams)

    net = AlphaZeroNet(board_area=42, num_actions=7, input_depth=2).to(mcts_hyperparams['device'])
    # net = DummyAlphaZeroNet(3, 16).to(mcts_hyperparams['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=training_hyperparams['lr'],
                                 weight_decay=training_hyperparams['l2_reg'])
    
    total_episodes = training_hyperparams['num_episodes']
    episodes_per_train = training_hyperparams['num_episodes_per_train']
    num_trains = int(total_episodes / episodes_per_train)

    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=total_episodes)
        for i in range(num_trains):
            # training_buffer.clear()
            for j in range(episodes_per_train):
                current_episode = i * episodes_per_train + j + 1
                progress.update(task, advance=1, description=f"[red]Training... (Episode {current_episode}/{total_episodes})")
                training_buffer.extend(run_episode(net, mcts_hyperparams))
            if len(training_buffer) > 10000:
                training_buffer = training_buffer[-10000:]
            train_network(net, optimizer, training_buffer, training_hyperparams)
            torch.save(net.state_dict(), f'model_{wandb.run.id if wandb.run else 0}.pth')

    if USE_WANDB:
        wandb.finish()


def train_network(network, optimizer, training_buffer, hyperparams: dict):
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
        print(
            f"[blue]Policy Loss:[/blue] {policy_loss.item():.4f}, [blue]Value Loss:[/blue] {value_loss.item():.4f}, [blue]Total Loss:[/blue] {loss.item():.4f}")

        # Log the losses using wandb
        if USE_WANDB:
            wandb.log({
                "Policy Loss": policy_loss.item(),
                "Value Loss": value_loss.item(),
                "Total Loss": loss.item()
            })

    network.to(torch.device('cpu'))
    network.eval()


def run_episode(network, hyperparams: dict):
    root = Node(None, None)
    board = np.zeros((6, 7))
    done = False
    states = []
    search_policies = []
    final_value = 0

    cur_player = 1
    while not done:
        states.append(np.copy(board))
        policy, actions = mcts_search(board, root, network, hyperparams)
        action_idx = np.random.choice(len(actions), p=policy)
        final_value, done = step(board, actions[action_idx])
        root = root.children[action_idx]
        complete_policy = np.zeros(7)
        complete_policy[actions] = policy
        search_policies.append(complete_policy)

        board *= -1
        if not done:
            cur_player *= -1

    values = np.ones(len(states))
    values[1::2] = -1
    values = list(values * final_value)
    return (zip(states, search_policies, values))


if __name__ == "__main__":
    train()

    # while not done:
    # policy, actions = mcts_search(board, root, network, hyperparams)
    # action_idx = np.random.choice(len(actions), p=policy)
    # final_value, done = step(board, actions[action_idx])
    # root = root.children[action_idx]
    # # print(board)

    # states.append(board)
    # complete_policy = np.zeros(7)
    # complete_policy[actions] = policy
    # search_policies.append(complete_policy)
