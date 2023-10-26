import torch
class AlphaZeroDataset(torch.utils.data.Dataset):
    def __init__(self, gamecontainer):
        self.gamecontainer = gamecontainer
        self.board_area = 6*7
        self.num_actions = 7
        self.input_depth = 8
        self.samples = []
    
    def init_from_replay():
        pass

    def __len__(self):
        return len(self.gamecontainer)
    
    def __getitem__(self, idx):
        return self.gamecontainer[idx]