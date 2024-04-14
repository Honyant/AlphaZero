from network import AlphaZeroNet
import torch
import numpy as np
from mcts import Node, mcts_search
from game import step
from rich.progress import Progress

mcts_hyperparams = {
        'iterations': 25,
        'c_puct': 1.0,
        'tau': 1,
        'device': torch.device('cpu')
        #'mps' if torch.backends.mps.is_available() else 
}

training_hyperparams = {
    'lr': 0.01,
    'l2_reg': 0.01,
    'batch_size': 64,
    'num_train_iter': 1,
    'num_episodes' : 100,
    'num_episodes_per_train': 10,
    'device': torch.device('cpu')

}

def train():
    training_buffer = []
    net = AlphaZeroNet(board_area=42, num_actions=7, input_depth=1).to(mcts_hyperparams['device'])
    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=training_hyperparams['num_episodes'])
        for _ in range(int(training_hyperparams['num_episodes']/training_hyperparams['num_episodes_per_train'])):
            training_buffer.clear()
            for _ in range(training_hyperparams['num_episodes_per_train']):
                progress.update(task, advance=1)
                starting_player = 1 if np.random.rand() < 0.5 else -1
                training_buffer.extend(run_episode(net, starting_player, mcts_hyperparams))
            train_network(net, training_buffer, training_hyperparams)
    torch.save(net.state_dict(), 'model.pth')

def train_network(network, training_buffer, hyperparams : dict):
    optimizer = torch.optim.Adam(network.parameters(), lr=hyperparams['lr'])
    for _ in range(hyperparams['num_train_iter']):
        indices = np.random.choice(len(training_buffer), hyperparams['batch_size'], replace=False)
        batch = [training_buffer[i] for i in indices]       #batch should be a list of tuples
        states, search_policies, values = [np.array(x) for x in zip(*batch)]
        states = torch.Tensor(states).to(hyperparams['device']).unsqueeze(1)
        search_policies = torch.Tensor(search_policies).to(hyperparams['device'])
        values = torch.Tensor(values).to(hyperparams['device'])
        
        optimizer.zero_grad()
        policy, value = network(states)
        policy_loss = -torch.sum(search_policies * torch.log(policy))
        value_loss = torch.sum((values - value)**2)
        regularization_loss = torch.sum(torch.stack([torch.sum(param**2) for param in network.parameters()]))
        loss = policy_loss + value_loss + hyperparams['l2_reg'] * regularization_loss
        loss.backward()
        optimizer.step()
    


def run_episode(network, starting_player, hyperparams : dict):
    root = Node(None, None)
    board = np.zeros((6,7))
    
    done = False
    
    states = []
    search_policies = []
    final_value = 0
    
    cur_player = starting_player
    
    while not done:
        policy, actions = mcts_search(board, root, network, hyperparams)
        action_idx = np.random.choice(len(actions), p=policy)
        final_value, done = step(board, actions[action_idx])
        root = root.children[action_idx]
        # print(board)
        
        states.append(board)
        complete_policy = np.zeros(7)
        complete_policy[actions] = policy
        search_policies.append(complete_policy)
        
        cur_player *= -1
        board *= -1
        
    values = list(np.ones(len(states)) * final_value * cur_player)
    return list(zip(states, search_policies, values))
    
if __name__ == "__main__":
    train()