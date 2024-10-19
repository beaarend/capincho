import random
import numpy
import torchvision.datasets as dset
from torch.utils.data import Dataset
from embeddingsLoader import COCODataset
import pandas
import torch


class CaptioningDataset(Dataset):
    def __init__(self, embeddings_path, text_only=True):
        if text_only:
            df = pandas.read_csv('datasets_torchvision/coco_2017/texts.csv')
            self.texts = df['texts'].values
        else:
            assert False, 'not implemented yet, use text_only=True'
            # data = dset.CocoCaptions(root=f'datasets_torchvision/coco_2017/train2017',
            #                          annFile=f'datasets_torchvision/coco_2017/annotations/captions_train2017.json', )
            # self.texts = []
            # for img, texts in data:
            #     self.texts.append(texts[:5])
            # self.texts = numpy.array(self.texts)

        self.embeddings = None
        self.text_only = text_only
        embeddings = COCODataset(embeddings_path)
        loader, indices = embeddings.get_loader(shuffle=False)

        for batch in loader:
            if text_only:
                embed = batch[1]
                embed = torch.reshape(embed, (embed.shape[0] * embed.shape[1], embed.shape[2]))
                embed = embed.detach()
            else:
                embed = batch[0].detach()

            if self.embeddings is None:
                self.embeddings = embed
            else:
                self.embeddings = numpy.concatenate((self.embeddings, embed))
        print(self.embeddings.shape, self.texts.shape)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        if self.text_only:
            return {'embeddings': torch.tensor(self.embeddings[index]), 'captions': self.texts[index]}
        else:
            text = self.texts[index]
            i = random.randint(0, 4)
            return {'embeddings': torch.tensor(self.embeddings[index]), 'captions': text[i]}

    def get_loader(self, shuffle=False, batch_size=400):
        indices = numpy.arange(len(self.texts))
        if shuffle:
            numpy.random.shuffle(indices)
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)
        return loader


if __name__ == '__main__':
    dataset = CaptioningDataset(f'embeddings/coco_contrastive_train.pkl', text_only=True)
    loader = dataset.get_loader()
    # for i in range(100):
    #     for batch in loader:
    #         print(len(batch['embeddings']), len(batch['captions']))




