import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
import argparse
from foundation_models import OpenCLIP, CLIP
import numpy as np


class TextLoader(Dataset):
    def __init__(self, data_path, has_embeddings=False, split='train'):

        if not has_embeddings:
            data = pd.read_excel(data_path)
            lim = int(0.9 * len(data))
            if split == 'train':
                data = data[:lim]
            elif split == 'val':
                data = data[lim:]
            self.texts = data['texts']
        else:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            lim = int(0.9 * len(data['image_embeddings']))
            if split == 'train':
                self.embeddings = data['image_embeddings'][:lim]
                self.texts = data['texts_embeddings'][:lim]
                self.image_ids = data['image_id'][:lim]
            elif split == 'val':
                self.embeddings = data['image_embeddings'][lim:]
                self.texts = data['texts_embeddings'][lim:]
                self.image_ids = data['image_id'][lim:]
        self.has_embeddings = has_embeddings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        if self.has_embeddings:
            embedding = torch.tensor(self.embeddings[index])
            return {
                'captions': self.texts[index],
                'image_embeddings': embedding,
                'image_id': self.image_ids[index]
            }

    def get_loader(self, batch_size=32):
        indices = np.arange(len(self.texts))
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)
        return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', '-p', type=str, required=True, help='path to texts xlsx')
    parser.add_argument('--output', '-o', type=str, required=True, help='output path')
    parser.add_argument('--model', '-m', type=str, required=True, help='model name', choices=['openclip', 'clip'])
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = TextLoader(args.path, has_embeddings=False)

    if args.model == 'openclip':
        model = OpenCLIP(device)
        model.load_model()

    else:
        model = CLIP(device)
        model.load_model()

    embeddings = []
    texts = []
    for batch in data.get_loader(batch_size=2):
        with torch.no_grad():
            embeddings += model.language_embedding(batch['captions']).detach().cpu().tolist()
            texts += batch['captions']

    new_dict = {'captions': texts, 'embeddings': embeddings}
    # print(embeddings)
    # print(texts)
    with open(args.output, 'wb') as f:
        pickle.dump(new_dict, f)
