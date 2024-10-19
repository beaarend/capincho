import argparse
import torch
import os
from pycocotools.coco import COCO
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pickle
import foundation_models
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def extract_features_geo(model, dataset='dataset/geoVQA.xlsx'):
    output = {'image_path': [], 'text': [], 'image_features': [], 'text_features': [], 'text_length': [], 'width': [],
              'height': []}
    df = pd.read_excel(dataset)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        path = row['image']
        image = Image.open(path)
        width, height = image.size
        caption = row['gt_text']
        with torch.no_grad():
            img_features = model.visual_embedding(row['image'])
            text_features = model.language_embedding(caption)

        output['image_features'].append(img_features.cpu())
        output['text_features'].append(text_features.cpu())
        output['text_length'].append(len(caption))
        output['width'].append(width)
        output['height'].append(height)
        output['image_path'].append(row['image'])
        output['text'].append(caption)

    with open('dataset/embeddings/clip_embeddings.pkl', 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='split to use')
    parser.add_argument('--model', choices=['openclip', 'clip', 'coca'], default='openclip', help='model to use')
    parser.add_argument('--save_path', type=str, default='embeddings/coco_train.pkl')
    args = parser.parse_args()

    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP}
    model = model_dict[args.model](device)
    model.load_model()

    coco = COCO(f'datasets_torchvision/coco_2017/annotations/captions_{args.split}2017.json')
    ids = coco.getImgIds()
    imgs = coco.loadImgs(ids)

    data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'texts_embeddings': []}
    for i, image in enumerate(tqdm(imgs)):
        data['image_name'].append(image['file_name'])
        data['image_id'].append(ids[i])
        img_embeds = model.visual_embedding('datasets_torchvision/coco_2017/{}2017/{}'.format(args.split,
                                                                                              image['file_name']))
        data['image_embeddings'].append(img_embeds.detach().cpu())

        ann = coco.loadAnns(coco.getAnnIds(ids[i]))
        texts = [e['caption'] for e in ann]
        text_embeds = model.language_embedding(texts)
        data['texts_embeddings'].append(text_embeds.detach().cpu())

    with open(args.save_path, 'wb') as f:
        pickle.dump(data, f)


