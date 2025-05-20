import torch
from torch.utils.data import DataLoader, TensorDataset

import os.path
import json
import pickle
from tqdm import tqdm
import argparse
import foundation_models
import numpy as np

import torch.nn as nn
from torch.optim import Adam

from embeddingsDataset import EmbeddingDataset
from dataLoader import DatasetHandler
from util import plot_curves, apply_lora, get_lora_parameters
from lora import LoRAWrapper

from earlyStopping import EarlyStopping

from util import dataset_path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def adapt_lora_embeddings_old(model, args, split, save_path, dataset):

#     if(dataset == 'rsicd'):
#         embed_path = f'embeddings/rsicd_nolora_{split}.pkl'
#         captions_json = os.path.join(dataset_path, 'RSICD/annotations', f'{split}_split.json')

#         handler = DatasetHandler(captions_json)
#         image_ids = handler.get_image_ids()

#         id_to_captions = {}
#         for img in handler.dataset['images']:
#             img_id = img['imgid']
#             sentences = img.get('sentences', [])
#             captions = [sent['raw'] for sent in sentences][:5]
#             if len(captions) < 5:
#                 captions += [""] * (5 - len(captions))
#             id_to_captions[img_id] = captions

#     if(dataset == 'coco'):
#         captions_json = os.path.join(dataset_path, 'COCO', 'annotations', f'captions_{split}2017.json')
#         embed_path = f'embeddings/coco/coco_{split}.pkl'

#         with open(captions_json, 'r') as f:
#             coco_data = json.load(f)

#         # image_id -> list of captions
#         id_to_captions = {}
#         for ann in coco_data['annotations']:
#             img_id = ann['image_id']
#             caption = ann['caption']
#             if img_id not in id_to_captions:
#                 id_to_captions[img_id] = []
#             id_to_captions[img_id].append(caption)

#         # pad/truncate to 5 captions each
#         for img_id in id_to_captions:
#             captions = id_to_captions[img_id]
#             if len(captions) < 5:
#                 captions += [""] * (5 - len(captions))
#             else:
#                 captions = captions[:5]
#             id_to_captions[img_id] = captions

#     embed_dataset = EmbeddingDataset(embed_path)
#     loader, indices = embed_dataset.get_loader(shuffle=False, batch_size=args.batch_size)

#     all_image_embeddings = []
#     all_text_embeddings = []
#     all_image_ids = []
#     all_image_names = []
#     all_captions = []

#     model.foundation.backbone.eval()

#     for batch in tqdm(loader, desc="Adapting embeddings"):
#         with torch.no_grad():
#             text_features = batch['texts_embeddings'].to(device, torch.float32)
#             B, N, D = text_features.shape
#             text_features_flat = text_features.view(B * N, 1, D)
#             text_proj, _ = model.textAdapter(text_features_flat, text_features_flat, text_features_flat)
#             text_proj = text_proj.squeeze(1)
#             text_proj = text_proj / text_proj.norm(dim=1, keepdim=True)
#             text_proj = text_proj.view(B, N, D)

#             image_features = batch['image_embeddings'].to(device, torch.float32).squeeze()
#             image_features = image_features.unsqueeze(1)
#             image_proj, _ = model.imageAdapter(image_features, image_features, image_features)
#             image_proj = image_proj.squeeze(1)
#             image_proj = image_proj / image_proj.norm(dim=1, keepdim=True)

#         all_image_embeddings.append(image_proj.cpu())
#         all_text_embeddings.append(text_proj.cpu())

#         # Convert image IDs to int
#         image_ids_int = [int(i) if isinstance(i, torch.Tensor) else i for i in batch['image_id']]
#         all_image_ids.extend(image_ids_int)

#         # Confirm image names are strings
#         for name in batch['image_name']:
#             assert isinstance(name, str), f"Invalid image_name type: {type(name)}"
#         all_image_names.extend(batch['image_name'])

#         # Fetch and validate captions
#         for img_id in image_ids_int:
#             captions = id_to_captions.get(img_id, [""] * 5)
#             assert isinstance(captions, list) and all(isinstance(c, str) for c in captions), \
#                 f"Invalid captions for image_id {img_id}: {captions}"
#             all_captions.append(captions)

#     data = {
#         'image_embeddings': torch.cat(all_image_embeddings, dim=0),
#         'texts_embeddings': torch.cat(all_text_embeddings, dim=0),
#         'image_id': all_image_ids,
#         'image_name': all_image_names,
#         'captions': all_captions
#     }

#     with open(save_path, 'wb') as f:
#         pickle.dump(data, f)

#     print(f"Saved adapted embeddings with captions to {save_path}")

def adapt_lora_embeddings(model, args, split, save_path, dataset):
    loaded_dataset = EmbeddingDataset(f'embeddings/{dataset}_nolora_{split}.pkl')
    loader, index = loaded_dataset.get_loader(shuffle=False, batch_size=args.batch_size)

    all_images = []
    all_texts = []
    all_ids = []
    all_names = []

    for batch in loader:
        images, _ = model.image_projection(batch['image_embeddings'])
        images = images.detach().cpu()

        texts, _ = model.text_projection(batch['texts_embeddings'])
        texts = texts.detach().cpu()

        all_images.extend(images)
        all_texts.extend(texts)
        all_ids.extend([int(i) for i in batch['image_id']])
        all_names.extend([str(name) for name in batch['image_name']])

    data = {
        'image_embeddings': all_images,
        'texts_embeddings': all_texts,
        'image_id': all_ids,
        'image_name': all_names,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved adapted embeddings to {save_path}")

def run_lora_training(model, args, save_path):

    model.backbone.train()
    list_lora_layers = apply_lora(args, model)

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in get_lora_parameters(model, bias='lora_only'): 
        param.requires_grad = True
 
    if(args.dataset == 'coco'):
        train_dataset = EmbeddingDataset('embeddings/coco/coco_train.pkl')
        val_dataset = EmbeddingDataset('embeddings/coco/coco_val.pkl')
    if(args.dataset == 'rsicd'):
        train_dataset = EmbeddingDataset('embeddings/rsicd_nolora_train.pkl')
        val_dataset = EmbeddingDataset('embeddings/rsicd_nolora_val.pkl')

    train_loader, train_indices = train_dataset.get_loader(shuffle=True, batch_size=args.batch_size)
    val_loader, val_indices = val_dataset.get_loader(shuffle=True, batch_size=args.batch_size)
    train_losses = []
    val_losses = []

    adapter_params = list(model.textAdapter.parameters()) + list(model.imageAdapter.parameters())
    lora_params = list(get_lora_parameters(model, bias='lora_only'))
    optim = torch.optim.Adam(lora_params + adapter_params + [model.logit_scale], lr=args.lr, weight_decay=1e-4)

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

        with torch.no_grad():
            model.logit_scale.data.clamp_(0, np.log(100))
        print(f"logit_scale raw: {model.logit_scale.item():.4f}, exp: {model.logit_scale.exp().item():.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('created directory', save_path)
        
    plot_curves(train_losses, val_losses, os.path.join(save_path, 'loss_plot.png'))
    log = {'training_loss': train_losses, 'validation_loss': val_losses}
    with open(os.path.join(save_path, 'loss_log.pkl'), 'wb') as f:
        pickle.dump(log, f)
    
    exit()

    if(args.dataset == 'coco'):
        new_train_embeddings = 'embeddings/coco/coco_lora_train_1.pkl'
        new_val_embeddings = 'embeddings/coco/coco_lora_val_1.pkl'
    if(args.dataset == 'rsicd'):
        new_train_embeddings = 'embeddings/rsicd_lora_train_3.pkl'
        new_val_embeddings = 'embeddings/rsicd_lora_val_3.pkl'

    adapt_lora_embeddings(model, args, 'train', new_train_embeddings, args.dataset)
    adapt_lora_embeddings(model, args, 'val', new_val_embeddings, args.dataset)

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
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--dataset', default='coco', choices=['coco', 'rsicd'], type=str)

    args = parser.parse_args()

    combinations = [
        {'position': 'all', 'params': ['q', 'k', 'v', 'o'], 'lr': 2e-4, 'dropout_rate': 0.2, 'r': 8, 'alpha': 16},
        {'position': 'all', 'params': ['q', 'k', 'v', 'o'], 'lr': 5e-5, 'dropout_rate': 0.2, 'r': 8, 'alpha': 8},
        # {'position': 'all', 'params': ['o'], 'lr': 1e-4, 'dropout_rate': 0.1, 'r': 4, 'alpha': 4},
        # {'position': 'all', 'params': ['k', 'v'], 'lr': 1e-4, 'dropout_rate': 0.3, 'r': 4, 'alpha': 8},
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

        param_str = f"pos_{args.position}_params_{'-'.join(args.params)}_lr_{args.lr}_r_{args.r}_drop_{args.dropout_rate}"
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