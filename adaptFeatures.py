import argparse
import os
import pickle
import json
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
    # single batch
    loader, indices = dataset.get_loader(batch_size=len(dataset), shuffle=False)
    for batch in loader:
        images = model.image_projection(batch['image_embeddings']).cpu()
        texts = model.text_projection(batch['texts_embeddings']).cpu()
        data = {'image_embeddings': images,
                'texts_embeddings': texts,
                'image_id': batch['image_id'],
                'image_name': batch['image_name']}

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-n', type=str, required=True, help='experiment name to load')
    parser.add_argument('--output', '-o', type=str, required=True, help='output file path')
    parser.add_argument('--embeddings', '-e', type=str, required=True, help='embeddings file path')
    args = parser.parse_args()

    with open(args.experiment, 'r') as f:
        config = json.load(f)

    logit_scale = config['logit_scale'] * torch.ones([])

    if config['adapter'] == 'contrastive':
        model = ContrastiveResidualAdapter(config['embedding_dim'], config['alpha'], logit_scale,
                                           config['learnable_alpha'])

    else:
        model = SigAdapter(config['embedding_dim'], config['alpha'], config['bias'],logit_scale,
                           config['multiple_positives'], config['use_bias'], )

    adapt_features(model,
                   checkpoint_path=config['checkpoint_path'],
                   save_path=args.output,
                   embeddings_path=args.embeddings)

