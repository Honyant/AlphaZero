import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return torch.sigmoid(self.fc2(y))
    
class AlphaZeroNet(nn.Module):
    def __init__(self, board_area, num_actions, input_depth, blocks=2, conv_channels=32, head_channels=32, policy_channels=2, value_channels=1):
        super().__init__()
        self.input_conv = InputConvBlock(input_depth, conv_channels)
        self.residual_tower = ResidualTower(conv_channels, blocks)
        self.policy_head = PolicyHead(head_channels, policy_channels, num_actions, board_area)
        self.value_head = ValueHead(head_channels, value_channels, board_area)

    def forward(self, x):
        y = self.input_conv(x)
        y = self.residual_tower(y)
        return self.policy_head(y), self.value_head(y)
    