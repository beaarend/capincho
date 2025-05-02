import torch
from torch.utils.data import DataLoader, TensorDataset

import os.path
import json
import pickle
from tqdm import tqdm
import argparse
import foundation_models
import os

import torch.nn as nn
from torch.optim import Adam

from embeddingsDataset import EmbeddingDataset
from util import plot_curves, apply_lora, get_lora_parameters
from lora import LoRAWrapper

from earlyStopping import EarlyStopping

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def run_lora_training(model, args, save_path):

    model.backbone.train()
    list_lora_layers = apply_lora(args, model)

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in get_lora_parameters(model, bias='lora_only'): 
        param.requires_grad = True

    for name, module in model.backbone.named_modules():
        if "transformer.layers" in name and any(str(i) in name for i in [10, 11]):  # last 2 layers
            for param in module.parameters():
                param.requires_grad = True
 
    train_dataset = EmbeddingDataset('embeddings/rsicd_nolora_train.pkl')
    val_dataset = EmbeddingDataset('embeddings/rsicd_nolora_val.pkl')
    train_loader, train_indices = train_dataset.get_loader(shuffle=False, batch_size=args.batch_size)
    val_loader, val_indices = val_dataset.get_loader(shuffle=False, batch_size=args.batch_size)
    train_losses = []
    val_losses = []

    adapter_params = list(model.textAdapter.parameters()) + list(model.imageAdapter.parameters())
    lora_params = list(get_lora_parameters(model, bias='lora_only'))
    optim = torch.optim.Adam(lora_params + adapter_params, lr=args.lr, weight_decay=1e-4)

    early_stopper = EarlyStopping(patience=20, minimal_improvement=0.001, objective='minimize', save_option='best', save_path=save_path +'/best_model.pt')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5, factor=0.1, verbose=True)

    for epoch in range(args.n_iters):
        train_loss = model.train_epoch(train_loader, optim)
        val_loss = model.val_epoch(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{args.n_iters}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        early_stopper.update(val_loss, model.backbone.state_dict())
        if early_stopper.stop:
            print("Early stopping triggered.")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('created directory', save_path)
        
    plot_curves(train_losses, val_losses, os.path.join(save_path, 'loss_plot.png'))
    log = {'training_loss': train_losses, 'validation_loss': val_losses}
    with open(os.path.join(save_path, 'loss_log.pkl'), 'wb') as f:
        pickle.dump(log, f)
    
    new_train_embeddings = 'embeddings/rsicd_lora_train.pkl'
    new_val_embeddings = 'embeddings/rsicd_lora_val.pkl'
    #adapt_lora_embeddings(model, args, 'train', new_train_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='split to use')
    parser.add_argument('--model', choices=['openclip', 'clip', 'coca'], default='openclip', help='model to use')
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1.25, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    # model_dict = {'coca': foundation_models.OpenCoCa,
    #               'clip': foundation_models.CLIP,
    #               'openclip': foundation_models.OpenCLIP,
    #               'capivara': foundation_models.Capivara}
    
    # model = model_dict[args.model](device)

    # model.load_model()

    # model = LoRAWrapper(model, encoder='both')
    # model.backbone.to(device)
    # run_lora_training(model, args, save_path='results_rsicd/training')
    # model.backbone.eval()

    # Define combinations
    combinations = [
        {'position': 'bottom', 'params': ['q'], 'lr': 2e-4, 'dropout_rate': 0.25, 'r': 2},
        {'position': 'mid',    'params': ['q', 'k'], 'lr': 1e-4, 'dropout_rate': 0.3, 'r': 4},
        {'position': 'up',     'params': ['q', 'k', 'v'], 'lr': 5e-5, 'dropout_rate': 0.1, 'r': 8},
    ]

    for combo in combinations:
        args = parser.parse_args([])  # Start fresh for each combination

        # Set shared/default args
        args.model = 'openclip'
        args.encoder = 'both'
        args.n_iters = 200
        args.batch_size = 32

        # Set from combination
        args.position = combo['position']
        args.params = combo['params']
        args.lr = combo['lr']
        args.dropout_rate = combo['dropout_rate']
        args.r = combo['r']

        # Build a unique directory name based on the combination
        param_str = f"pos_{args.position}_params_{'-'.join(args.params)}_lr_{args.lr}_r_{args.r}_drop_{args.dropout_rate}"
        save_path = os.path.join('results_rsicd/training', param_str)
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