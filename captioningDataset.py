import pickle
import random
import numpy
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from dataLoader import DatasetHandler
import os
import torch

from util import dataset_path

class CaptioningDataset(Dataset):
    def __init__(self, embeddings_path, text_only=True):
        self.text_only = text_only

        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        self.embeddings = []
        self.captions = []

        assert len(self.embeddings) == len(self.captions), "Mismatch between embeddings and captions length"

        # captions
        if 'val' in embeddings_path:
            #coco = COCO(f'datasets_torchvision/coco_2017/annotations/captions_val2017.json')
            #coco = COCO(f'{dataset_path}/COCO/annotations/captions_val2017.json')
            loaded_dataset = DatasetHandler(os.path.join(dataset_path,'RSICD', 'annotations', 'val_split.json'))

        else:
            #coco = COCO(f'datasets_torchvision/coco_2017/annotations/captions_train2017.json')
            #coco = COCO(f'{dataset_path}/COCO/annotations/captions_train2017.json')
            loaded_dataset = DatasetHandler(os.path.join(dataset_path, 'RSICD','annotations', 'train_split.json'))

        for embed in embeddings['image_id']:
            #ann = coco.loadAnns(coco.getAnnIds(embed))
            ann = loaded_dataset.load_annotations(loaded_dataset.get_annotation_ids(embed))

            texts = [e['raw'] for e in ann]
            if text_only:
                self.captions += texts[:5]
            else:
                self.captions.append(texts[:5])

        # embeddings
        if text_only:
            if not torch.is_tensor(embeddings['texts_embeddings']):
                self.embeddings = torch.stack(embeddings['texts_embeddings'], dim=0)
            else:
                self.embeddings = embeddings['texts_embeddings']

            n, c, d = self.embeddings.shape
            self.embeddings = self.embeddings.view(n*5, d)

        else:
            self.embeddings = embeddings['image_embeddings']

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
    #dataset = CaptioningDataset(f'embeddings/coco_openclip_train.pkl', text_only=True)
    #dataset = CaptioningDataset(f'embeddings/coco_train.pkl', text_only=False)
    dataset = CaptioningDataset(f'embeddings/rsicd_lora_val.pkl', text_only=True)
    # print(len(dataset))
    # print(dataset[:]['embeddings'])
    # print(dataset['embeddings'])
    # print(dataset[:]['captions'])
    # loader = dataset.get_loader()
    # print(dataset[:]['captions'])
    # for batch in loader:
        # print(len(batch['captions']))
    #     print(batch['embeddings'].shape, len(batch['captions']))
    # with open('embeddings/coco_openclip_train.pkl', 'rb') as f:
    #     embeddings = pickle.load(f)
    #     print(embeddings['texts_embeddings'])





