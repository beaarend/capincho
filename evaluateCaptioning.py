import argparse
import random
import torch
from textLoader import TextLoader
from dataLoader import DatasetHandler
from decoder import model_from_json
from tqdm import tqdm
from util import dataset_path
import os
import pickle

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('-e', '--embeddings', type=str, help='path to embeddings file', default='embeddings/rsicd_lora_val.pkl')
#     parser.add_argument('-m', '--model', type=str, default='results_rsicd/decoder/experiment.json',
#                         help='path to experiment json file')
#     parser.add_argument('-s', '--split', type=str, default='val', choices=['train', 'val'],
#                         help='split to load for evaluation')
#     parser.add_argument('--random_seed', type=int, default=777, help='random seed for qualitative evaluation')
#     parser.add_argument('--num_images', '-n', type=int, default=10, help='number of images to evaluate')
#     args = parser.parse_args()

#     #data = TextLoader(args.embeddings, has_embeddings=True, split='train')
#     data = DatasetHandler(args.embeddings)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model_from_json(args.model, device)
#     model.eval()
#     random.seed(args.random_seed)
#     for i in tqdm([random.randint(0, len(data)) for i in range(args.num_images)]):
#         # print(data[i]['image_embeddings'].shape)
#         generated = model.caption(data[i]['image_embeddings'], max_tokens=200, )
#         print(f"id: {data[i]['image_id']}")
#         if type(data[i]['captions']) is list:
#             print('ORIGINAL: ' + data[i]['captions'][0])
#         else:
#             print(f"ORIGINAL: {data[i]['captions']}")

#         print('GENERATED: ' + generated[0])

def load_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate caption generation')
    parser.add_argument('-e', '--embeddings', type=str, default='embeddings/rsicd_lora_train.pkl')
    parser.add_argument('-m', '--model', type=str, default='results_rsicd/decoder/experiment.json')
    parser.add_argument('-s', '--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--random_seed', type=int, default=777)
    parser.add_argument('--num_images', '-n', type=int, default=10)
    args = parser.parse_args()

    random.seed(args.random_seed)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_json(args.model, device)
    model.eval()

    # Load annotations and embeddings
    handler = DatasetHandler(os.path.join(dataset_path,'RSICD', 'annotations', f'{args.split}_split.json'))
    embeddings_data = load_embeddings(args.embeddings)

    image_ids = embeddings_data['image_id']
    image_embeddings = embeddings_data['image_embeddings']

    # Sample indices
    indices = [random.randint(0, len(image_ids) - 1) for _ in range(args.num_images)]

    for i in tqdm(indices):
        img_id = image_ids[i]
        embed = image_embeddings[i]
        #embed_tensor = torch.tensor(embed).unsqueeze(0).to(device)

        generated = model.caption(embed, max_tokens=200)

        # Load original caption from annotation JSON
        original_captions = handler.load_annotations([img_id])
        original_text = original_captions[0]["raw"] if original_captions else "[No original caption found]"

        print(f"\nid: {img_id}")
        print(f"ORIGINAL: {original_text}")
        print(f"GENERATED: {generated[0] if isinstance(generated, list) else generated}")