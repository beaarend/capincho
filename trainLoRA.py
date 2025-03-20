import torch
from torch.utils.data import DataLoader, TensorDataset

import os.path
import time
import pickle
from tqdm import tqdm
import argparse
import foundation_models

import torch.nn as nn
from torch.optim import Adam

from embeddingsDataset import COCODataset
from util import plot_curves, apply_lora, get_lora_parameters
from lora import LoRAdapter

import EarlyStopping as EarlyStopping

from peft import get_peft_model, LoraConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "")

def run_lora_training(save_path, batch_size, embeddings_path, model, epochs, patience, delta):
    
    train_dataset = COCODataset(embeddings_paths[0])
    val_dataset = COCODataset(embeddings_paths[1])
    train_loader, train_indices = train_dataset.get_loader(shuffle=False, batch_size=args.batch_size)
    val_loader, val_indices = val_dataset.get_loader(shuffle=False, batch_size=args.batch_size)

    if patience < 0:
        patience = epochs

    es = EarlyStopping(patience=patience, minimal_improvement=delta, objective='minimize', save_option=save_option)
    training_losses = []
    validation_losses = []

    optimizer = torch.optim.AdamW(get_lora_parameters(model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)

    print(f'training LORAAAAAAA {os.path.basename(save_path)}')
    time.sleep(1)

    # TODO CONECTAR MODEL LORAADAPTER COM O MODEL AQUI

    for i in tqdm(range(epochs)):
        training_loss = model.train_epoch(train_loader, optimizer)
        validation_loss = model.val_epoch(val_loader)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        model_dict = {'epoch': i,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': training_losses[-1]
                      }
        es.update(validation_loss, model_dict)
        if es.stop:
            break

    torch.save(es.model_to_save(), os.path.join(save_path, 'checkpoint.pt'))
    plot_curves(training_losses, validation_losses, os.path.join(save_path, 'loss_plot.png'))
    log = {'training_loss': training_losses, 'validation_loss': validation_losses}
    with open(os.path.join(save_path, 'loss_log.pkl'), 'wb') as f:
        pickle.dump(log, f)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='openclip', choices=['openclip', 'clip', 'coca'],
    #                     help='foundation model')
    # parser.add_argument('--adapter', type=str, default='contrastive', choices=['contrastive', 'sig', 'mixer'],
    #                     help='adapter type')
    # parser.add_argument('--alpha', type=float, default=0.3, help='residual learning rate')
    # parser.add_argument('--bias', type=float, default=-10., help='logit bias, sig adapter')
    # parser.add_argument('--embeddings', type=str, required=True,
    #                     help='training embeddings path')
    # parser.add_argument('--use_bias', action='store_true', help='use logit bias in sig adapter', default=False)
    # parser.add_argument('--multiple_positives', action='store_true',
    #                     help='use multiple positives per batch in sig adapter', default=False)
    # parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    # parser.add_argument('--embedding_dim', type=int, default=768, help='embedding dimension')
    # parser.add_argument('--learnable_alpha', action='store_true', help='learnable alpha', default=False)
    # parser.add_argument('--save_path', type=str, required=True, help='path to save outputs')
    # parser.add_argument('--patience', type=int, default=-1, help='early stopping patience, '
    #                                                              'negative value means no early stopping')
    # parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    # parser.add_argument('--best', action='store_true', help='restore best model if using early stopping', default=False)
    # parser.add_argument('--delta', type=float, help='minimal improvement for early stopping', default=0.01,)
    # parser.add_argument('--epochs', type=int, default=200, help='number training of epochs')

    #parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    #parser.add_argument('--root_path', type=str, default='')
    #parser.add_argument('--dataset', type=str, default='dtd')
    #parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--model', type=str, default='openclip', choices=['openclip', 'clip', 'coca'],
                         help='foundation model')
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    #parser.add_argument('--embeddings', type=str, required=True, help='training embeddings path')
    
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    #parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    #parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')

    args = parser.parse_args()

    train_embeddings_path = 'embeddings/coco_train.pkl'
    val_embeddings_path = 'embeddings/coco_val.pkl'
    embeddings_paths = [train_embeddings_path, val_embeddings_path]

    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)
    #     print('created directory', args.save_path)

    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP}

    foundation = model_dict[args.model](device)
    foundation.load_model()

    logit_scale = foundation.backbone.logit_scale

    model = model_dict[args.model](device)
    model.load_model()

    list_lora_layers = apply_lora(args, model)
    print(list_lora_layers)
    model.cuda()

    run_lora_training(args.save_path, args.batch_size, embeddings_paths, model, args.epochs, args.lr, args.patience, args.delta, args.best)

    optimizer = torch.optim.AdamW(get_lora_parameters(model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_iters, eta_min=1e-6)

    



    # training loop


