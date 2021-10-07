import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

def read_data(embeddings_path='X.npy', labels_path='y.npy', reports_path=None):
    X = np.load(embeddings_path, allow_pickle=True)
    y = np.load(labels_path, allow_pickle=True)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    if reports_path is None:
        return TensorDataset(X, y)
    reports_used = np.load(reports_path)
    return TensorDataset(X, y), reports_used


def create_dataloader(dataset, test_size=0.1):
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)
    train_dataloader = DataLoader(train_dataset,
                                        sampler = RandomSampler(train_dataset),
                                        batch_size = 128)
    test_dataloader = DataLoader(test_dataset,
                            sampler = RandomSampler(test_dataset),
                            batch_size = 128)
                            
    return train_dataloader, test_dataloader