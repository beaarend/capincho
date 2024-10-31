import pandas as pd
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class TextLoader(Dataset):
    def __init__(self, data_path, model):
        data = pd.read_excel(data_path)
        self.texts = data['texts']
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokens = tokenizer(self.texts, return_tensors='pt')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {'labels': self.texts[index], 'tokens': self.tokens[index]}
