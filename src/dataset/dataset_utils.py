import torch
from torch.utils.data import DataLoader, Dataset, random_split


def split_data(data, splits, batch_size, random_seed=42):
    assert sum(splits) == 1

    n = len(data)
    val_size = int(splits[1] * n)
    test_size = int(splits[2] * n)
    train_size = n - val_size - test_size
    train_set, val_set, test_set = random_split(data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))

    train_set = DataLoader(dataset=train_set,
                               batch_size=batch_size,
                               shuffle=True)
    val_set = DataLoader(dataset=val_set,
                             batch_size=batch_size,
                             shuffle=False)
    test_set = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False)
    
    return train_set, val_set, test_set