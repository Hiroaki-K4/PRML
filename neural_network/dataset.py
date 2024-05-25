import numpy as np
import torch
from torch.utils.data import Dataset


class MixedDensityNetworkDataset(Dataset):
    def __init__(self, input_arr, label_arr):
        input_arr = input_arr.astype(np.float32)
        label_arr = label_arr.astype(np.float32)
        self.input_x = torch.from_numpy(input_arr).clone()
        self.label = torch.from_numpy(label_arr).clone()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        input_x = self.input_x[idx]
        label = self.label[idx]
        return input_x, label
