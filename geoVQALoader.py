import os.path
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class GeoVQADataset(Dataset):
    def __init__(self, file_path):
        self.images = []
        self.texts = []
        assert os.path.exists(file_path), f"File {file_path} not found"

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        assert 'image_features' in data.keys(), "Missing 'image_features'"
        assert 'text_features' in data.keys(), "Missing 'text_features'"
        assert len(data['image_features']) == len(data['text_features']), "image_features and text_features must have the same length"
        assert len(data['image_features']) > 0, "image_features is empty"
        assert len(data['text_features']) > 0, "text_features is empty"

        for i in range(len(data['image_features'])):
            img = data['image_features'][i]
            txt = data['text_features'][i]
            if len(data['text_features'][i].shape) == 3:
                # blip variant
                self.texts.append(txt[:, 0, :])
                similarity = (img @ txt[:, 0, :].t())
                self.images.append(img[0, torch.argmax(similarity, dim=1).item(), :])

            elif len(data['image_features'][i].shape) == 2:
                # clip variant
                self.images.append(img[0, :])
                self.texts.append(txt[0, :])

            elif len(data['image_features'][i].shape) == 1:
                # resized embeddings
                self.images.append(img)
                self.texts.append(txt)

    def __getitem__(self, index):
        return self.images[index],  self.texts[index]

    def __len__(self):
        return len(self.images)

    def get_loaders(self, random_seed=59, batch_size=64, train_ratio=0.7, shuffle=True):
        size = len(self)
        indices = list(range(size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        split = int(size * train_ratio)
        train_indices = indices[:split]
        test_indices = indices[split:]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=test_sampler)
        return train_loader, test_loader, train_indices, test_indices


