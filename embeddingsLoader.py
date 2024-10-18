import os.path
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, path, n_captions=5):
        assert os.path.exists(path), '{} does not exist'.format(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.captions = data['text_features']
            self.images = data['image_features']
            self.n = n_captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.captions[index][:self.n]

    def get_loader(self, shuffle=True, batch_size=400):
        indices = np.arange(len(self.images))
        if shuffle:
            np.random.shuffle(indices)
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)
        return loader, indices


class EmbeddingDataset(Dataset):
    def __init__(self, file_path, k=0):
        self.images = []
        self.labels = []
        assert os.path.exists(file_path), f"File {file_path} not found"
        # file_path = 'datasets_torchvision/embeddings/flowers_ViTL_train.pkl'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        assert 'image_features' in data.keys(), "Missing 'image_features'"
        assert 'labels' in data.keys(), "Missing 'labels'"
        assert len(data['image_features']) == len(data['labels']), "image_features and label must have the same length"
        assert len(data['image_features']) > 0, "image_features is empty"
        assert len(data['labels']) > 0, "label is empty"

        if k == 0:
            for i in range(len(data['image_features'])):
                self.images.append(data['image_features'][i])
                self.labels.append(data['labels'][i])
        else:
            df = pd.DataFrame.from_dict(data)
            categories = df['labels'].unique()
            for category in categories:
                samples = df[df['labels'] == category]
                samples = samples.sample(frac=1)
                samples = samples.iloc[:k, :]

                for i, row in samples.iterrows():
                    self.images.append(row['image_features'])
                    self.labels.append(row['labels'])

    def __getitem__(self, index):
        return self.images[index][0],  self.labels[index]

    def __len__(self):
        return len(self.images)

    def get_loaders(self, random_seed=59, batch_size=64, shuffle=True, train_ratio=0.7):
        indices = list(range(len(self.images)))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        split = int(len(indices) * train_ratio)
        train_indices = indices[:split]
        val_indices = indices[split:]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=test_sampler)
        return train_loader, val_loader, train_indices, val_indices


