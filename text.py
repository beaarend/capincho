import torch
from embeddingsDataset import EmbeddingDataset
# checkpoint = torch.load('results/adapter2/checkpoint.pt')
# print(checkpoint.keys())

# print(checkpoint['model_state_dict'].keys())

# dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'])
# odict_keys(['logit_scale', 'imageAdapter.residual', 'imageAdapter.model.0.weight', 'imageAdapter.model.0.bias', 
#             'imageAdapter.model.2.weight', 'imageAdapter.model.2.bias', 'textAdapter.residual', 'textAdapter.model.0.weight', 
#             'textAdapter.model.0.bias', 'textAdapter.model.2.weight', 'textAdapter.model.2.bias'])

# checkpoint = torch.load('/mnt/d/new_results/capinchoDecoder/checkpoint.pt')
# print(checkpoint.keys())

# print(checkpoint['model_state_dict'].keys())

# checkpoint = torch.load('results/lora/checkpoint.pt')
# print(checkpoint.keys())

# print(checkpoint['model_state_dict'].keys())


embed_lora = EmbeddingDataset('embeddings/coco/coco_lora_train_VERDADEIRO.pkl')
embed_base = EmbeddingDataset('embeddings/coco/coco_train.pkl')

# Stack image embeddings into tensors
images_lora = torch.stack(embed_lora.images)   # shape: (N, D)
images_base = torch.stack(embed_base.images)   # shape: (N, D)

# Normalize both for cosine similarity
images_lora = images_lora / images_lora.norm(dim=-1, keepdim=True)
images_base = images_base / images_base.norm(dim=-1, keepdim=True)

print(embed_lora.images[0][0:5])
print(embed_base.images[0][0:5])

texts_lora = torch.cat([torch.stack(t) for t in embed_lora.text_embeddings])
texts_base = torch.cat([torch.stack(t) for t in embed_base.text_embeddings])

texts_lora = texts_lora / texts_lora.norm(dim=-1, keepdim=True)
texts_base = texts_base / texts_base.norm(dim=-1, keepdim=True)

cos_sim_texts = torch.nn.functional.cosine_similarity(texts_lora, texts_base, dim=-1)
print(f"Mean cosine similarity between LORA and BASE text embeddings: {cos_sim_texts.mean():.6f}")

# Cosine similarity between each corresponding pair
cos_sim = torch.nn.functional.cosine_similarity(images_lora, images_base, dim=-1)

# Average similarity
print(f"Mean cosine similarity between LORA and BASE image embeddings: {cos_sim.mean():.6f}")