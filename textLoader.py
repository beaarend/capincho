import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
import argparse
from foundation_models import OpenCLIP, CLIP


class TextLoader(Dataset):
    def __init__(self, data_path):
        data = pd.read_excel(data_path)
        self.texts = data['texts']
        self.tokens = data['embeddings']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {'labels': self.texts[index], 'tokens': self.tokens[index]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', '-p', type=str, required=True, help='path to texts xlsx')
    parser.add_argument('--output', '-o', type=str, required=True, help='output path')
    parser.add_argument('--model', '-m', type=str, required=True, help='model name', choices=['openclip', 'clip'])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    data = pd.read_excel(args.path)
    texts = data['texts']
    if args.model == 'openclip':
        model = OpenCLIP(device)
        embeddings = model.language_embedding(texts)
    else:
        model = CLIP(device)
        embeddings = model.language_embedding(texts)

    new_dict = {'texts': texts, 'embeddings': embeddings}
    with open(args.output, 'wb') as f:
        pickle.dump(new_dict, f)
