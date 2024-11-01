import pickle
import pandas as pd
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import argparse


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
    parser.add_argument('--path', type=str, required=True, help='path to texts xlsx')
    parser.add_argument('--model', type=str, required=True, help='hugginface model')
    parser.add_argument('--output', type=str, required=True, help='output path')
    args = parser.parse_args()

    data = pd.read_excel(args.path)
    texts = data['texts']
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokens = tokenizer(texts, return_tensors='pt')

    new_dict = {'texts': texts, 'tokens': tokens}
    with open(args.output, 'wb') as f:
        pickle.dump(new_dict, f)
