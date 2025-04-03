import argparse
import random
import torch
from textLoader import TextLoader
from decoder import model_from_json
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--embeddings', type=str, required=True, help='path to embeddings file')
    parser.add_argument('-m', '--model', type=str, default='experiments/checkpoint.json', required=True,
                        help='path to experiment json file')
    parser.add_argument('-s', '--split', type=str, default='val', choices=['train', 'val'],
                        help='split to load for evaluation')
    parser.add_argument('--random_seed', type=int, default=777, help='random seed for qualitative evaluation')
    parser.add_argument('--num_images', '-n', type=int, default=10, help='number of images to evaluate')
    args = parser.parse_args()

    data = TextLoader(args.embeddings, has_embeddings=True, split='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_json(args.model, device)
    model.eval()
    random.seed(args.random_seed)
    for i in tqdm([random.randint(0, len(data)) for i in range(args.num_images)]):
        # print(data[i]['image_embeddings'].shape)
        generated = model.caption(data[i]['image_embeddings'], max_tokens=200, )
        print('id: ' + data[i]['image_id'])
        if type(data[i]['captions']) is list:
            print('ORIGINAL: ' + data[i]['captions'][0])
        else:
            print('ORIGINAL: ' + data[i]['captions'])

        print('GENERATED: ' + generated[0])
