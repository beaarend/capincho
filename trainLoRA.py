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

    # Set optimizer to optimize LoRA parameters
    optim = torch.optim.Adam(get_lora_parameters(model, bias='lora_only'), lr=args.lr)

    # Training loop
    for epoch in range(args.n_iters):
        train_loss = model.train_epoch(train_loader, optim)
        val_loss = model.val_epoch(val_loader)

        print(f"Epoch {epoch + 1}/{args.n_iters}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
