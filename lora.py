import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from projectionHeads import PlainMultiHeadAttention
from lora_layers import PlainMultiheadAttentionLoRA
from torch.nn import MultiheadAttention
import numpy as np
from embeddingsDataset import EmbeddingDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LoRAWrapper:
    def __init__(self, foundation, encoder, embed_dim=768):
        self.foundation = foundation
        self.encoder = encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def __getattr__(self, name):
        foundation = object.__getattribute__(self, "foundation") 
        if hasattr(foundation, name):
            return getattr(foundation, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, batch):

        image = batch['image'].to(device)
        text_input_ids = batch['text'].to(device)
        # attention_mask = batch['attention_mask'].to(device)

        # with torch.no_grad():
        #     image_features = self.foundation.encode_image(image)
        #     text_features = self.foundation.encode_text(text_input_ids, attention_mask=attention_mask)

        image_features = self.foundation.backbone.encode_image(image)
        text_features = self.foundation.backbone.encode_text(text_input_ids)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        if self.encoder == 'both':
            cosine_similarity = logit_scale * image_features @ text_features.T
            return cosine_similarity
            
        else:
            raise ValueError("NOT IMPLEMENTED USE BOTH" )
    
    def train_epoch(self, train_loader, optimizer):

        self.foundation.backbone.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            
            logits = self.forward(batch)  # shape [batch_size, batch_size]
            
            batch_size = batch['image'].size(0)
            targets = torch.arange(batch_size, device=logits.device)
            
            loss_img_to_text = nn.CrossEntropyLoss()(logits, targets)
            loss_text_to_img = nn.CrossEntropyLoss()(logits.T, targets)
            loss = loss_img_to_text + loss_text_to_img
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.detach().cpu().item())
        
        return np.mean(epoch_losses)

    def val_epoch(self, val_loader):
        self.foundation.backbone.eval()  # or self.foundation.backbone.eval()
        epoch_losses = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                logits = self.forward(batch)  
                batch_size = batch['image'].size(0)
                targets = torch.arange(batch_size, device=logits.device)

                loss_img_to_text = nn.CrossEntropyLoss()(logits, targets)
                loss_text_to_img = nn.CrossEntropyLoss()(logits.T, targets)
                loss = loss_img_to_text + loss_text_to_img
                epoch_losses.append(loss.cpu().item())

                if i == 0:
                    # Print some cosine similarities for the first batch
                    print("\n📊 Cosine Similarities (top-left 5x5 matrix):")
                    print(logits[:5, :5].cpu().numpy())

                    print("\n✅ Matching Pairs (diagonal):")
                    for j in range(min(5, batch_size)):
                        print(f"Pair {j}: {logits[j, j].item():.4f}")

                    print("\n❌ Non-Matching Pairs (off-diagonal):")
                    for j in range(min(5, batch_size)):
                        k = (j + 1) % batch_size
                        print(f"Image {j} vs Text {k}: {logits[j, k].item():.4f}")

        return np.mean(epoch_losses)
