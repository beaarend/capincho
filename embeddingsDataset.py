import os.path
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, path, n_captions=5):
        assert os.path.exists(path), '{} does not exist'.format(path)
        self.captions = []
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for i in data['texts_embeddings']:
                self.captions.append(i[:n_captions])
            self.images = data['image_embeddings']
            self.image_id = data['image_id']
            self.image_name = data['image_name']
            self.n = n_captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        payload = {'image_id': self.image_id[index],
                   'image_name': self.image_name[index],
                   'image_embeddings': self.images[index],
                   'texts_embeddings': self.captions[index]}
        return payload

    def get_loader(self, shuffle=False, batch_size=400):
        indices = np.arange(len(self.images))
        if shuffle:
            np.random.shuffle(indices)
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)
        return loader, indices


if __name__ == '__main__':
    dataset = COCODataset('embeddings/coco_openclip_val.pkl')
    print(dataset[0]['image_embeddings'].shape, dataset[0].keys(), len(dataset[:]['image_embeddings']),
          len(dataset[:]['texts_embeddings']))

    loader, indices = dataset.get_loader()
    for batch in loader:
        print(batch['image_embeddings'].shape, batch['texts_embeddings'].shape)