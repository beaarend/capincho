import torch
from torch.utils.data import DataLoader, TensorDataset

import os.path
import json
import pickle
from tqdm import tqdm
import argparse
import foundation_models
import numpy as np
import math
import time

import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

from open_clip import tokenize

from embeddingsDataset import EmbeddingDataset
from dataLoader import DatasetHandler, RSICDDataset
from util import plot_curves, apply_lora, get_lora_parameters, mark_only_lora_as_trainable
from lora import LoRAWrapper

from earlyStopping import EarlyStopping

from util import dataset_path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]  # raw text strings

    # Use OpenCLIP tokenizer
    tokenized = tokenize(texts) 

    return {
        "image": images,
        "text": tokenized,              
        "attention_mask": None,         
    }

def run_lora_training(model, args, save_path):

    if args.dataset == 'coco':
        pass
    elif args.dataset == 'rsicd':
        train_image_path = "/mnt/d/new_data/RSICD/rsicd_train"
        train_annotation_file = "/mnt/d/new_data/RSICD/annotations/train_split.json"
        val_image_path = "/mnt/d/new_data/RSICD/rsicd_val"
        val_annotation_file = "/mnt/d/new_data/RSICD/annotations/val_split.json"

    model.backbone.train()

    list_lora_layers = apply_lora(args, model)

    mark_only_lora_as_trainable(model, bias='lora_only')   

    model.logit_scale.requires_grad = True
 
    train_handler = DatasetHandler(annotation_file=train_annotation_file, dataset_type=args.dataset)
    val_handler = DatasetHandler(annotation_file=val_annotation_file, dataset_type=args.dataset)

    train_dataset = RSICDDataset(train_handler, image_dir=train_image_path)
    val_dataset = RSICDDataset(val_handler, image_dir=val_image_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    lora_params = list(get_lora_parameters(model, bias='lora_only'))

    early_stopper = EarlyStopping(patience=5, minimal_improvement=0.01, objective='minimize', save_option='best', save_path=save_path +'/best_model.pt')
    optimizer = torch.optim.AdamW(lora_params + [model.logit_scale], weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_iters, eta_min=1e-6)

    warmup_steps = max(10, int(0.04 * args.n_iters))  # e.g. 5% of total iters or at least 10 epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # linear warmup from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # cosine annealing from 1 to 0
            progress = float(current_step - warmup_steps) / float(max(1, args.n_iters - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses = []
    val_losses = []

    model.foundation.backbone.to(device)

    for epoch in range(args.n_iters):
        print(f"Starting epoch {epoch + 1}/{args.n_iters}...")
        start_time = time.time()  # Start timer
        
        train_loss = model.train_epoch(train_loader, optimizer)
        val_loss = model.val_epoch(val_loader)

        epoch_duration = time.time() - start_time

        scheduler.step()

        print(f"Epoch {epoch + 1}/{args.n_iters} finished in {epoch_duration:.2f} seconds")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        early_stopper.update(val_loss, model.backbone.state_dict())
        if early_stopper.stop:
            print("Early stopping triggered.")
            break

        print(f"logit_scale raw: {model.logit_scale.item():.4f}, exp: {model.logit_scale.exp().item():.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    print("Training complete.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('created directory', save_path)
        
    plot_curves(train_losses, val_losses, os.path.join(save_path, 'loss_plot.png'))
    log = {'training_loss': train_losses, 'validation_loss': val_losses}
    with open(os.path.join(save_path, 'loss_log.pkl'), 'wb') as f:
        pickle.dump(log, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='val', help='split to use')
    parser.add_argument('--model', choices=['openclip', 'clip', 'coca'], default='openclip', help='model to use')
    parser.add_argument('--backbone', default='ViT-B/32', type=str)
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='vision')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1.25, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--dataset', default='rsicd', choices=['coco', 'rsicd'], type=str)

    args = parser.parse_args()

    combinations = [
        {'position': 'up', 'params': ['q', 'k', 'v', 'o'], 'lr': 5e-5, 'dropout_rate': 0.2, 'r': 2, 'alpha': 0.5},
        # {'position': 'all', 'params': ['q', 'k', 'v', 'o'], 'lr': 1e-5, 'dropout_rate': 0.1, 'r': 4, 'alpha': 1.25},
        # {'position': 'up', 'params': ['o'], 'lr': 5e-5, 'dropout_rate': 0.3, 'r': 4, 'alpha': 0.75},
        # {'position': 'mid', 'params': ['o'], 'lr': 2e-4 , 'dropout_rate': 0.1, 'r': 4, 'alpha': 0.50},
        # {'position': 'half-up', 'params': ['q', 'k', 'v', 'o'], 'lr': 1e-5, 'dropout_rate': 0.3, 'r': 8, 'alpha': 0.25},
    ]


    for combo in combinations:
        args = parser.parse_args([])  

        args.model = 'openclip'
        args.encoder = 'both'
        args.n_iters = 300
        args.batch_size = 32

        args.position = combo['position']
        args.params = combo['params']
        args.lr = combo['lr']
        args.dropout_rate = combo['dropout_rate']
        args.r = combo['r']

        param_str = f"pos_{args.position}_params_{'-'.join(args.params)}_lr_{args.lr}_r_{args.r}_drop_{args.dropout_rate}_alpha_{args.alpha}"
        save_path = os.path.join(f'results_{args.dataset}', 'training', param_str)
        os.makedirs(save_path, exist_ok=True)

        print(f"\nRunning combination: {param_str}")

        model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP,
                  'capivara': foundation_models.Capivara}
    
        model = model_dict[args.model](device)
        model.load_model()
        model = LoRAWrapper(model, encoder=args.encoder)
        model.backbone.to(device)
        run_lora_training(model, args, save_path)
        model.backbone.eval()