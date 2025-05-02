import torch
from torch.utils.data import DataLoader, TensorDataset

import os.path
import json
import pickle
from tqdm import tqdm
import argparse
import foundation_models

import torch.nn as nn
from torch.optim import Adam

from embeddingsDataset import EmbeddingDataset
from util import plot_curves, apply_lora, get_lora_parameters
from lora import LoRAWrapper

from earlyStopping import EarlyStopping

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_lora_training(model, args):

    model.backbone.train()
    list_lora_layers = apply_lora(args, model)

    for param in model.backbone.parameters():
        param.requires_grad = True

    # for param in get_lora_parameters(model, bias='lora_only'): 
    #     param.requires_grad = True

    # for name, module in model.backbone.named_modules():
    #     if "transformer.layers" in name and any(str(i) in name for i in [10, 11]):  # last 2 layers
    #         for param in module.parameters():
    #             param.requires_grad = True
 
    train_dataset = EmbeddingDataset('embeddings/rsicd_nolora_train.pkl')
    val_dataset = EmbeddingDataset('embeddings/rsicd_nolora_val.pkl')
    train_loader, train_indices = train_dataset.get_loader(shuffle=False, batch_size=args.batch_size)
    val_loader, val_indices = val_dataset.get_loader(shuffle=False, batch_size=args.batch_size)

    adapter_params = list(model.textAdapter.parameters()) + list(model.imageAdapter.parameters())
    lora_params = list(get_lora_parameters(model, bias='lora_only'))
    optim = torch.optim.Adam(lora_params + adapter_params, lr=args.lr)
    # optim = torch.optim.Adam(get_lora_parameters(model, bias='lora_only'), lr=args.lr)

    # Training loop
    for epoch in range(args.n_iters):
        train_loss = model.train_epoch(train_loader, optim)
        val_loss = model.val_epoch(val_loader)

        print(f"Epoch {epoch + 1}/{args.n_iters}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    new_train_embeddings = 'embeddings/rsicd_lora_train.pkl'
    new_val_embeddings = 'embeddings/rsicd_lora_val.pkl'
    adapt_lora_embeddings(model, args, 'train', new_train_embeddings)

def adapt_lora_embeddings(model, args, type, save_path):

    if (type == 'val'):
        dataset = EmbeddingDataset('embeddings/rsicd_nolora_val.pkl')
        loader, indices = dataset.get_loader(shuffle=False, batch_size=args.batch_size)
    else:
        dataset = EmbeddingDataset('embeddings/rsicd_nolora_train.pkl')
        loader, indices = dataset.get_loader(shuffle=False, batch_size=args.batch_size)

    for batch in loader:
        images = model.image_projection(batch['image_embeddings']).detach().cpu()
        texts = model.text_projection(batch['texts_embeddings']).detach().cpu()

        data = {'image_embeddings': images,
                'texts_embeddings': texts,
                'image_id': dataset[:]['image_id'],
                'image_name': dataset[:]['image_name'],}

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='split to use')
    parser.add_argument('--model', choices=['openclip', 'clip', 'coca'], default='openclip', help='model to use')
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP,
                  'capivara': foundation_models.Capivara}
    
    model = model_dict[args.model](device)

    model.load_model()

    model = LoRAWrapper(model, encoder='both')
    model.backbone.to(device)

    run_lora_training(model, args)