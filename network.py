import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ResidualTower(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

    def forward(self, x):
        return self.blocks(x)


class InputConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class PolicyHead(nn.Module):
    def __init__(self, in_channels, middle_channels, num_actions, board_area):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, middle_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(middle_channels)
        self.fc = nn.Linear(board_area * middle_channels, num_actions)

    def forward(self, x):
        y = F.relu(self.bn(self.conv(x)))
        y = y.view(y.size(0), -1)
        return F.softmax(self.fc(y), dim=1)


class ValueHead(nn.Module):
    def __init__(self, in_channels, middle_channels, board_area, intermediate_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, middle_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(middle_channels)
        self.fc1 = nn.Linear(board_area * middle_channels, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, 1)

    def forward(self, x):
        y = F.relu(self.bn(self.conv(x)))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        return torch.tanh(self.fc2(y))


class AlphaZeroNet(nn.Module):
    def __init__(self, board_area, num_actions, input_depth, blocks=5, conv_channels=4, head_channels=4, policy_channels=4, value_channels=4):
        super().__init__()
        self.input_conv = InputConvBlock(input_depth, conv_channels)
        self.residual_tower = ResidualTower(conv_channels, blocks)
        self.policy_head = PolicyHead(head_channels, policy_channels, num_actions, board_area)
        self.value_head = ValueHead(head_channels, value_channels, board_area)

    def forward(self, x):
        y = self.input_conv(x)
        y = self.residual_tower(y)
        return self.policy_head(y), self.value_head(y)


def convert_array_torch(arr):
    if type(arr) == np.ndarray:
        arr = torch.tensor(arr)
    result = torch.zeros((arr.shape[0], 2, 6, 7), dtype=torch.float32).to(arr.device)
    
    result[:, 0] = (arr == 1).squeeze(1)
    result[:, 1] = (arr == -1).squeeze(1)
    
    return result

def get_policy_and_value(net: AlphaZeroNet, board: np.array, hyperparams: dict) -> int:
    # policy, value = net(torch.Tensor(board.flatten()).unsqueeze(0).to(hyperparams['device']))
    policy, value = net(convert_array_torch(torch.Tensor(board).unsqueeze(0).unsqueeze(0).to(hyperparams['device'])))
    policy = policy.detach().cpu().numpy().flatten()
    value = value.detach().cpu().numpy().flatten().item()
    policy = (policy * (1 - np.abs(board[0])))
    policy = policy[policy != 0] / np.sum(policy)
    return policy, value


class DummyAlphaZeroNet(nn.Module):
    def __init__(self, n_hidden, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(n_hidden):
            layers.append(nn.LazyLinear(n_hidden))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.value_head = nn.LazyLinear(1)
        self.policy_head = nn.LazyLinear(7)

    def forward(self, x):
        z = self.net(x)
        v = self.value_head(z)
        p = self.policy_head(z)

        return F.softmax(p), F.tanh(v)