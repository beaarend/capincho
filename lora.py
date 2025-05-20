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

        temp_text_mha = MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        temp_image_mha = MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.textAdapter = PlainMultiheadAttentionLoRA(temp_text_mha).to(device)
        self.imageAdapter = PlainMultiheadAttentionLoRA(temp_image_mha).to(device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def __getattr__(self, name):
        foundation = object.__getattribute__(self, "foundation") 
        if hasattr(foundation, name):
            return getattr(foundation, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, batch):
        if self.encoder == 'text' or self.encoder == 'both':
            text_features = batch['texts_embeddings'].to(device, torch.float32)
            # c = random.randint(0, text_features.shape[1] - 1)
            # text_features = text_features[:, c, :]
            # text_features = text_features.unsqueeze(1) 
            text_features = text_features.mean(dim=1, keepdim=True)
            text_features, _ = self.textAdapter(text_features, text_features, text_features)
            text_features = text_features.squeeze(1)  
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if self.encoder == 'vision' or self.encoder == 'both':
            image_features = batch['image_embeddings'].to(device, torch.float32).squeeze()  
            image_features = image_features.unsqueeze(1)
            image_features, _ = self.imageAdapter(image_features, image_features, image_features)
            image_features = image_features.squeeze(1)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        cosine_similarity = logit_scale * image_features @ text_features.T
        # cosine_similarity = self.foundation.backbone.logit_scale * image_features @ text_features.T
        return cosine_similarity
    
    def image_projection(self, embeddings):
        self.foundation.backbone.eval()
        return self.imageAdapter(embeddings.to(device, torch.float32), embeddings.to(device, torch.float32), embeddings.to(device, torch.float32))

    def text_projection(self, embeddings):
        self.foundation.backbone.eval()
        return self.textAdapter(embeddings.to(device, torch.float32), embeddings.to(device, torch.float32), embeddings.to(device, torch.float32))
    
    def train_epoch(self, train_loader, optim):
        self.foundation.backbone.train()
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
        self.foundation.backbone.eval()
        epoch_losses = []

        for batch in val_loader:
            logits = self.forward(batch)
            targets = torch.arange(len(batch['image_embeddings'])).to(device)
            i_loss = nn.CrossEntropyLoss()(logits, targets)
            t_loss = nn.CrossEntropyLoss()(logits.T, targets)
            loss = i_loss + t_loss
            epoch_losses.append(loss.detach().cpu())

        for i, batch in enumerate(val_loader):
            logits = self.forward(batch)
            targets = torch.arange(len(batch['image_embeddings'])).to(device)
            i_loss = nn.CrossEntropyLoss()(logits, targets)
            t_loss = nn.CrossEntropyLoss()(logits.T, targets)
            loss = i_loss + t_loss
            epoch_losses.append(loss.detach().cpu())

            if i == 0:
                with torch.no_grad():
                    text_features = batch['texts_embeddings'].to(device, torch.float32)
                    c = random.randint(0, text_features.shape[1] - 1)
                    text_features = text_features[:, c, :].unsqueeze(1)
                    text_features, _ = self.textAdapter(text_features, text_features, text_features)
                    text_features = text_features.squeeze(1)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

                    image_features = batch['image_embeddings'].to(device, torch.float32).squeeze().unsqueeze(1)
                    image_features, _ = self.imageAdapter(image_features, image_features, image_features)
                    image_features = image_features.squeeze(1)
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)

                    cos_sim = image_features @ text_features.T

                    print("\n📊 Cosine Similarities (top-left 5x5 matrix):")
                    print(cos_sim[:5, :5].cpu().numpy())

                    print("\n✅ Matching Pairs (diagonal):")
                    for j in range(min(5, cos_sim.shape[0])):
                        print(f"Pair {j}: {cos_sim[j, j].item():.4f}")

                    print("\n❌ Non-Matching Pairs (off-diagonal):")
                    for j in range(min(5, cos_sim.shape[0])):
                        k = (j + 1) % cos_sim.shape[0]
                        print(f"Image {j} vs Text {k}: {cos_sim[j, k].item():.4f}")

        return np.mean(epoch_losses)
    
