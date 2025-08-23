import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from sklearn.cluster import KMeans
from parallel.parallel_funcs import print_wise
from alignment.utils import get_usual_transform, get_transform_to_show


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



def multistage_kmeans(
    X,
    n_clusters_per_level,
    **kmeans_args
):
    """
    Perform multistage KMeans clustering.

    Args:
        X (torch.Tensor): Input data on CUDA.
        n_clusters_per_level (list): List of cluster counts for each stage.
        kmeans_args (dict, optional): Arguments to pass to kmeans_gpu.

    Returns:
        centers_list (list): List of centers at each stage.
    """
    if kmeans_args is None:
        kmeans_args = {
            'init': 'k-means++',
            'max_iter': 10000,
            'tol': 1e-6,
            'n_init': 20
        }


    centers_list = []
    labels_list = []
    print_wise(f"Levels: {len(n_clusters_per_level)} | Clusters per level: {n_clusters_per_level}")
    for level, size in enumerate(n_clusters_per_level):
        print_wise(f"--- Level {level+1}/{len(n_clusters_per_level)}: Clustering into {size} clusters ---")
        if level == 0:
            data = X
        else:
            data = centers_list[level - 1]
        print_wise(f"Data shape: {data.shape}")
        kmeans = KMeans(
            n_clusters=size,
            **kmeans_args
        )
        kmeans.fit(data)  
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        centers_list.append(centers)
        labels_list.append(labels)
    return centers_list, labels_list


def assign_clusters_in_batches(X, C, batch_size=1000):
    """
    Assign each point in X to its nearest centroid in C using Euclidean distance,
    computed in batches to avoid memory overload.

    Args:
        X: (N, D) array of data points
        C: (K, D) array of centroids
        batch_size: number of points per batch

    Returns:
        labels: (N,) array of cluster indices
    """
    N = X.shape[0]
    labels = np.empty(N, dtype=np.int32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = X[start:end]   # shape (B, D)
        # Compute distances (B, K)
        dists = np.linalg.norm(batch[:, None, :] - C[None, :, :], axis=2)
        labels[start:end] = np.argmin(dists, axis=1)

    return labels


def sample_cluster_wise(final_labels, sample_per_cluster):
    representatives = []
    for label in np.unique(final_labels):
        idxs = np.where(final_labels == label)[0]
        representatives.extend(np.random.choice(idxs, sample_per_cluster, replace=False))
    representatives = np.array(representatives)
    return representatives


def subset_loader(imagenet_val_path, idx, batch_size, shuffle=True, to_show=False):
    if to_show==True:
        transform = get_transform_to_show()
    else:
        transform = get_usual_transform()
    # end

    loader = DataLoader(
        Subset(datasets.ImageFolder(imagenet_val_path, transform=transform), idx),
        batch_size=batch_size,
        num_workers=1,
        shuffle=shuffle,
        pin_memory=True,
        timeout=500,
    )
    return loader
    
