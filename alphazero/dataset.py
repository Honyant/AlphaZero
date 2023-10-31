import torch
import numpy as np
from alphazero.gamecontainer import GameContainer
class AlphaZeroDataset(torch.utils.data.Dataset):
    def __init__(self, gamecontainer):
        self.gamecontainer = gamecontainer
        self.board_area = 6*7
        self.num_actions = 7
        self.input_depth = 8
        self.samples = []
    
    def init_from_replay(self):
        #get random samples from the gamecontainer:
        input_samples = []
        target_policy_samples = []
        target_value_samples = []

        for i in range(zip(self.gamecontainer.state_history, self.gamecontainer.child_visits, self.gamecontainer.value_history)):
            input_samples.append(i[0])
            target_policy_samples.append(i[1])
            target_value_samples.append(i[2])
        
        #convert to tensors:
        input_samples = torch.from_numpy(np.array(input_samples)).float()
        target_policy_samples = torch.from_numpy(np.array(target_policy_samples)).float()
        target_value_samples = torch.from_numpy(np.array(target_value_samples)).float()   

        #create dataset:
        self.samples = torch.utils.data.TensorDataset(input_samples, target_policy_samples, target_value_samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]