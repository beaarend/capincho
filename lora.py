import random
import torch
import torch.nn as nn
from projectionHeads import PlainMultiHeadAttention
import numpy as np
from embeddingsDataset import COCODataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LoRAdapter(nn.Module):
    def __init__(self, initial_logit_scale, encoder):
        self.encoder = encoder
        self.logit_scale = nn.Parameter(initial_logit_scale)

        #lora things here

    def forward(self, batch):
        if self.encoder == 'text' or self.encoder == 'both':
            text_features = batch['texts_embeddings'].to(device, torch.float32)
            c = random.randint(0, text_features.shape[1]-1)
            text_features = text_features[:, c, :]
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
        if self.encoder == 'vision' or self.encoder == 'both':
            image_features = batch['image_embeddings'].to(device, torch.float32).squeeze()
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
    
        cosine_similarity = self.logit_scale * image_features @ text_features.T
        return cosine_similarity

    def image_projection(self, embeddings):
        self.eval()
        return self.imageAdapter(embeddings.to(device, torch.float32))

    def text_projection(self, embeddings):
        self.eval()
        return self.textAdapter(embeddings.to(device, torch.float32))

    def train_epoch(self, train_loader, optim):
        self.train()
        epoch_losses = []
        for batch in train_loader:
            optim.zero_grad()
            logits = self.forward(batch)
            targets = torch.arange(len(batch['image_embeddings'])).to(device)
            i_loss = nn.CrossEntropyLoss()(logits, targets)
            t_loss = nn.CrossEntropyLoss()(logits.T, targets)
            loss = i_loss + t_loss
            loss.backward()
            optim.step()
            epoch_losses.append(loss.detach().cpu())
        return np.mean(epoch_losses)

    def val_epoch(self, val_loader):
        self.eval()
        epoch_losses = []

        for batch in val_loader:
            logits = self.forward(batch)
            targets = torch.arange(len(batch['image_embeddings'])).to(device)
            i_loss = nn.CrossEntropyLoss()(logits, targets)
            t_loss = nn.CrossEntropyLoss()(logits.T, targets)
            loss = i_loss + t_loss
            epoch_losses.append(loss.detach().cpu())
        return np.mean(epoch_losses)
