import torch
import torchvision.datasets as dset
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import foundation_models
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


def extract_features_torchvision(model, data, save_path):
    output = {'image_features': [], 'labels': [], }
    for image, label in tqdm(data):
        output['labels'].append(label)
        image = model.vision_preprocess(image).unsqueeze(0).to(device)
        image_feature = model.backbone.encode_image(image)
        features = image_feature.detach().cpu()
        output['image_features'].append(features)

    with open(save_path, 'wb') as f:
        pickle.dump(output, f)


def extract_features_coco(model, data, save_path):
    output = {'image_features': [], 'text_features': [], }
    for image, captions in tqdm(data):
        image = model.vision_preprocess(image).unsqueeze(0).to(device)
        # print(image.shape, )
        image_features = model.backbone.encode_image(image)
        image_features = image_features.detach().cpu()
        output['image_features'].append(image_features)

        text_features = model.language_embedding(captions)
        text_features = text_features.detach().cpu()
        output['text_features'].append(text_features)

    with open(save_path, 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    # print(open_clip.list_pretrained())
    model = foundation_models.OpenCoCa(device)
    model.load_model()

    for split in ['val', 'train']:
        # data = dset.StanfordCars(root='datasets_torchvision/stanford_cars', split=split)
        # data = dset.FGVCAircraft(root='datasets_torchvision/fgvc_aircraft/', split=split, annotation_level='variant')
        # data = dset.Flowers102(root='datasets_torchvision/flowers102', split=split)

        data = dset.CocoCaptions(root=f'datasets_torchvision/coco_2017/{split}2017',
                                 annFile=f'datasets_torchvision/coco_2017/annotations/captions_{split}2017.json', )

        extract_features_coco(model, data, f'datasets_torchvision/embeddings/coco_COCA'
                                           f'_{split}.pkl')


