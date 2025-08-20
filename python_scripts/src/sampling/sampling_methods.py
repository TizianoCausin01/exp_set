import numpy as np
from torch.utils.data import Dataset


def random_sets(max_val, n_sets, batch_size):
    if max_val < n_sets*batch_size:
        raise ValueError("Not enough elements to avoid replacement")
    tot_idx = np.random.choice(np.arange(max_val), size = n_sets*batch_size)
    set_list = []
    for set_idx in range(0, n_sets*batch_size, batch_size):
        curr_idx = tot_idx[set_idx:set_idx+batch_size] # without -1 because python excludes the last num while indexing
        set_list.append(curr_idx)
    return set_list




"""
class IndexedDataset

example usage:

imagenet_val_path = f"{paths['data_path']}/imagenet/val"
base_dataset = datasets.ImageFolder(imagenet_val_path, transform=transform)

dataset = IndexedDataset(base_dataset)

loader = DataLoader(dataset, batch_size=3, shuffle=True)

# Now each batch has (images, labels, indices)
for images, labels, indices in loader:
    print(images.shape)   # [32, 3, 224, 224]
    print(labels.shape)   # [32]
    print(indices)  # [32]
    break
"""

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # e.g. ImageFolder or CIFAR10

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx   # add the index to the output

    def __len__(self):
        return len(self.dataset)
