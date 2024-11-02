import pickle
import pandas as pd
from torch.utils.data import Dataset
import open_clip
import argparse
import clip


class TextLoader(Dataset):
    def __init__(self, data_path):
        data = pd.read_excel(data_path)
        self.texts = data['texts']
        self.tokens = data['tokens']

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

    data = pd.read_excel(args.path)
    texts = data['texts']
    if args.model == 'openclip':
        tokenizer = open_clip.get_tokenizer('ViT-L/14')
        tokens = tokenizer(texts)
    else:
        tokens = clip.tokenize(texts)

    new_dict = {'texts': texts, 'tokens': tokens}
    with open(args.output, 'wb') as f:
        pickle.dump(new_dict, f)
