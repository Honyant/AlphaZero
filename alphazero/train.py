import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from alphazero.model import AlphaZeroNet
from alphazero.dataset import AlphaZeroDataset
from alphazero.gamecontainer import GameContainer


# Set up hyperparameters
def train(gamecontainer, batch_size=32, num_epochs=2, learning_rate=0.001):

    # Set up the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = AlphaZeroDataset(gamecontainer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = AlphaZeroNet(board_area=dataset.board_area, num_actions=dataset.num_actions, input_depth=dataset.input_depth).to(device)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for i, (states, policies, values) in enumerate(dataloader):
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            optimizer.zero_grad()
            policy_pred, value_pred = model(states)
            policy_loss = torch.nn.functional.cross_entropy(policy_pred, policies)
            value_loss = torch.nn.functional.mse_loss(value_pred, values)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Step {i}, Batch {i}, Loss: {loss.item()}")


gamecontainer = GameContainer()
#gamecontainer.init_from_replay()
train(gamecontainer)