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

from embeddingsDataset import COCODataset
from util import plot_curves, apply_lora, get_lora_parameters
from lora import LoRAWrapper

from earlyStopping import EarlyStopping

from peft import get_peft_model, LoraConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "")

def run_lora_training(save_path, batch_size, embeddings_path, model, epochs, lr, patience, delta, save_option, iters):
    
    train_dataset = COCODataset(embeddings_paths[0])
    val_dataset = COCODataset(embeddings_paths[1])
    train_loader, train_indices = train_dataset.get_loader(shuffle=False, batch_size=batch_size)
    val_loader, val_indices = val_dataset.get_loader(shuffle=False, batch_size=batch_size)

    if patience < 0:
        patience = epochs

    es = EarlyStopping(patience=patience, minimal_improvement=delta, objective='minimize', save_option=save_option)
    training_losses = []
    validation_losses = []

    optimizer = torch.optim.AdamW(get_lora_parameters(model), weight_decay=1e-2, betas=(0.9, 0.999), lr=lr)

    for i in tqdm(range(epochs)):

        model.foundation.backbone.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= iters:
                break

            optimizer.zero_grad()
            loss = model.train_epoch([batch], optimizer)
            epoch_loss += loss

        epoch_loss /= min(iters, len(train_loader))
        training_losses.append(epoch_loss)

        #training_loss = model.train_epoch(train_loader, optimizer)
        model.foundation.backbone.eval()
        validation_loss = model.val_epoch(val_loader)

        #training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        model_dict = {'epoch': i,
                      'model_state_dict': model.foundation.backbone.state_dict(),
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
    parser.add_argument('--patience', type=int, default=-1, help='early stopping patience, '
                                                                  'negative value means no early stopping')
    parser.add_argument('--best', action='store_true', help='restore best model if using early stopping', default="best")
    parser.add_argument('--delta', type=float, help='minimal improvement for early stopping', default=0.01,)
    parser.add_argument('--epochs', type=int, default=200, help='number training of epochs')

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
    parser.add_argument('--save_path', default='results/lora2', help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')

    args = parser.parse_args()

    train_embeddings_path = 'embeddings/coco_train.pkl'
    val_embeddings_path = 'embeddings/coco_val.pkl'
    embeddings_paths = [train_embeddings_path, val_embeddings_path]

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('created directory', args.save_path)

    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP}

    foundation = model_dict[args.model](device)
    foundation.load_model()
    foundation = LoRAWrapper(foundation, args.encoder)

    list_lora_layers = apply_lora(args, foundation)

    run_lora_training(args.save_path, args.batch_size, embeddings_paths, foundation, args.epochs, args.lr, args.patience, args.delta, args.best, args.n_iters)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.save_path, 'checkpoint.pt')
    result_dict['logit_scale'] = foundation.backbone.logit_scale.detach().cpu().item()

    with open(os.path.join(args.save_path, 'experiment.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)



