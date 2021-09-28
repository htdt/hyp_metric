import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import collections


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=np.int)
    return labels_to_indices


def safe_random_choice(input_data, size):
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace).tolist()


class UniqueClassSempler(Sampler):
    def __init__(self, labels, m_per_class, rank=0, world_size=1, seed=0):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = sorted(list(self.labels_to_indices.keys()))
        self.m_per_class = m_per_class
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return (len(self.labels) // self.world_size) * self.m_per_class

    def __iter__(self):
        idx_list = []
        g = torch.Generator()
        g.manual_seed(self.seed * 10000 + self.epoch)
        idx = torch.randperm(len(self.labels), generator=g).tolist()
        size = len(self.labels) // self.world_size
        idx = idx[size * self.rank : size * (self.rank + 1)]
        for i in idx:
            t = self.labels_to_indices[self.labels[i]]
            idx_list += safe_random_choice(t, self.m_per_class)
        return iter(idx_list)

    def set_epoch(self, epoch):
        self.epoch = epoch
