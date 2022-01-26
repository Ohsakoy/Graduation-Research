import torch.utils.data as Data
import torch


class NoisyDataset(Data.Dataset):
    def __init__(self, data, targets,  target_transform=None):

        self.data = data
        self.targets = targets
        self.target_transform = target_transform
        

    def __getitem__(self, index):

        img, label = self.data[index], self.targets[index]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label, index)

    def __len__(self):

        return len(self.data)
