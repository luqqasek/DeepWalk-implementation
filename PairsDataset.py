import torch
from torch.utils.data import Dataset


class PairsDataset(Dataset):

    def __init__(self, walk):
        # nodes to encode
        self.input = [pair[0] for pair in walk]
        # context node to predict
        self.target = [pair[1] for pair in walk]

    def __getitem__(self, index):
        # index needs to be tensor to one hot encode it
        input_idx = torch.tensor(self.input[index])
        # index needs to be tensor for backpropagation reasons
        target_idx = torch.tensor(self.target[index])
        return input_idx, target_idx

    def __len__(self):
        # return the number of samples
        return len(self.input)
    