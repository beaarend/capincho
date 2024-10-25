import pickle
import random
import numpy
import torchvision.datasets as dset
from torch.utils.data import Dataset
from embeddingsDataset import COCODataset
import pandas
import torch


class CaptioningDataset(Dataset):
    def __init__(self, embeddings_path, text_only=True):
        self.text_only = text_only
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        self.embeddings = []
        if text_only:
            self.captions = []
            for e in embeddings['captions']:
                self.captions += e[:5]

            for e in embeddings['texts_embeddings']:
                self.embeddings += e[:5, :]

        else:
            self.embeddings = embeddings['image_embeddings']
            self.captions = embeddings['captions']
        print(len(self.embeddings), len(self.captions))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        if self.text_only:
            return {'embeddings': self.embeddings[index].unsqueeze(0), 'captions': self.captions[index]}
        else:
            r = random.randint(0, 4)
            return {'embeddings': self.embeddings[index], 'captions': self.captions[index][r]}

    def get_loader(self, shuffle=False, batch_size=32):
        indices = numpy.arange(len(self.captions))
        if shuffle:
            numpy.random.shuffle(indices)
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)
        return loader


if __name__ == '__main__':
    dataset = CaptioningDataset(f'embeddings/coco_openclip_adapter_val.pkl', text_only=True)
    # print(len(dataset))
    # print(len(dataset[:]['embeddings']))
    # print(dataset[:]['captions'])
    loader = dataset.get_loader()
    # print(dataset[:]['captions'])
    for batch in loader:
        # print(len(batch['captions']))
        print(batch['embeddings'].shape)






