import argparse
import os
import pickle
import time
import torch
from adapters import ContrastiveResidualAdapter, DynamicContrastiveResidualAdapter, SigAdapter
from tqdm import tqdm
from torch.optim import Adam
import foundation_models
from util import plot_curves
from geoVQALoader import GeoVQADataset
from embeddingsLoader import COCODataset
from earlyStopping import EarlyStopping
device = torch.device("cuda" if torch.cuda.is_available() else "")


def adapt_features(model,
                   checkpoint_path='checkpoints/contrastive/clip_residual_MPT.pt',
                   embeddings_path='datasets_torchvision/embeddings/coco_ViTL_val.pkl',
                   save_path='datasets_torchvision/embeddings/coco_MPT.pkl',):

    assert os.path.exists(checkpoint_path), f'No checkpoint found at {checkpoint_path}'
    assert os.path.exists(embeddings_path), f'No embedding found at {embeddings_path}'

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    dataset = COCODataset(embeddings_path)
    loader, indices = dataset.get_loader(batch_size=len(dataset), shuffle=False)

    for batch in loader:
        images = model.image_projection(batch['image_embeddings']).cpu()
        texts = model.text_projection(batch['texts_embeddings']).cpu()
        results = {'image_embeddings': images,
                   'texts_embeddings': texts,
                   'image_id': batch['image_id'],
                   'image_name': batch['image_name']}

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    print('done!')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--experiment', type=str, default='experiments/coco_contrastive.json')
    args = parse.parse_args()

    adapt_features(model,
                   checkpoint_path=f'checkpoints/contrastive/OpenCLIP_residual_adapter_0.3.pt',
                   save_path=f'embeddings/coco_contrastive_val.pkl',
                   dataset_path=f'embeddings/coco_openclip_val.pkl')

