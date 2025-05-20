import torch
import pickle
from embeddingsDataset import EmbeddingDataset

embed_rsicd_lora = EmbeddingDataset('embeddings/rsicd_lora_train_3.pkl')
embed_rsicd_base = EmbeddingDataset('embeddings/rsicd_nolora_train.pkl')

embed_coco_lora = EmbeddingDataset('embeddings/coco/coco_lora_train_VERDADEIRO.pkl')
embed_coco_base = EmbeddingDataset('embeddings/coco/coco_train.pkl')

# # Load embeddings and convert them to tensors
# if isinstance(embed_lora.images, torch.Tensor):
#     images_lora = embed_lora.images  # Already a tensor
# else:
#     images_lora = torch.stack(embed_lora.images)  # Stack if it's a list of tensors

# if isinstance(embed_base.images, torch.Tensor):
#     images_base = embed_base.images  # Already a tensor
# else:
#     images_base = torch.stack(embed_base.images)  # Stack if it's a list of tensors

# # Normalize embeddings
# images_lora = images_lora / images_lora.norm(dim=-1, keepdim=True)
# images_base = images_base / images_base.norm(dim=-1, keepdim=True)

# # Calculate cosine similarity in batches
# batch_size = 64  # Adjust this batch size depending on your available memory
# num_batches = (images_lora.size(0) + batch_size - 1) // batch_size  # To handle the last smaller batch

# cos_sim_sum = 0.0  # To accumulate cosine similarities
# num_elements = 0  # To accumulate the number of elements for averaging

# for i in range(num_batches):
#     start_idx = i * batch_size
#     end_idx = min((i + 1) * batch_size, images_lora.size(0))  # Ensure last batch is handled properly

#     batch_images_lora = images_lora[start_idx:end_idx]
#     batch_images_base = images_base[start_idx:end_idx]

#     # Calculate cosine similarity for the batch
#     cos_sim_batch = torch.nn.functional.cosine_similarity(batch_images_lora, batch_images_base, dim=-1)

#     # Update the sum of cosine similarities and count of elements
#     cos_sim_sum += cos_sim_batch.sum().item()
#     num_elements += cos_sim_batch.size(0)

# # Compute and print the average cosine similarity
# mean_cos_sim = cos_sim_sum / num_elements
# print(f"Mean cosine similarity between LORA and BASE image embeddings: {mean_cos_sim:.6f}")
print("Keys in RSICD LORA dataset:", embed_rsicd_lora.__dict__.keys())
print("Keys in RSICD BASE dataset:", embed_rsicd_base.__dict__.keys())
print("Keys in COCO LORA dataset:", embed_coco_lora.__dict__.keys())
print("Keys in COCO BASE dataset:", embed_coco_base.__dict__.keys())

# # Check the shape of the images and text embeddings
print("Shape of RSICD LORA images:", embed_rsicd_lora.images.shape)
print("Shape of RSICD LORA text embeddings:", len(embed_rsicd_lora.text_embeddings))
print("images in RSICD LORA dataset:", embed_rsicd_lora.images[:1])
print("texts in RSICD LORA dataset:", embed_rsicd_lora.text_embeddings[:1])
print("Image IDs in RSICD LORA dataset:", embed_rsicd_lora.image_id[:5])
print("image names in RSICD LORA dataset:", embed_rsicd_lora.image_name[:5])

print("Shape of RSICD BASE images:", embed_rsicd_base.images.shape)
print("Shape of RSICD BASE text embeddings:", len(embed_rsicd_base.text_embeddings))
print("images in RSICD BASE dataset:", embed_rsicd_base.images[:1])
print("texts in RSICD BASE dataset:", embed_rsicd_base.text_embeddings[:1])
print("Image IDs in RSICD BASE dataset:", embed_rsicd_base.image_id[:5])
print("image names in RSICD BASE dataset:", embed_rsicd_base.image_name[:5])
# print("Shape of COCO LORA images:", embed_coco_lora.images.shape)
# print("Shape of COCO LORA text embeddings:", len(embed_coco_lora.text_embeddings))
# print("Image IDs in COCO LORA dataset:", embed_coco_lora.image_id[:5])
# print("Shape of COCO BASE images:", embed_coco_base.images.shape)
# print("Shape of COCO BASE text embeddings:", len(embed_coco_base.text_embeddings))
# print("Image IDs in COCO BASE dataset:", embed_coco_base.image_id[:5])

# Keys in RSICD LORA dataset: dict_keys(['images', 'text_embeddings', 'image_id', 'image_name'])
# Keys in RSICD BASE dataset: dict_keys(['images', 'text_embeddings', 'image_id', 'image_name'])
# Keys in COCO LORA dataset: dict_keys(['images', 'text_embeddings', 'image_id', 'image_name'])
# Keys in COCO BASE dataset: dict_keys(['images', 'text_embeddings', 'image_id', 'image_name'])
# Shape of RSICD LORA images: torch.Size([30, 1, 768])
# Shape of RSICD LORA text embeddings: 30
# Image IDs in RSICD LORA dataset: tensor([10878, 10879, 10880, 10881, 10882])
# Shape of RSICD BASE images: torch.Size([8734, 1, 768])
# Shape of RSICD BASE text embeddings: 8734
# Image IDs in RSICD BASE dataset: [0, 1, 2, 3, 4]
# Shape of COCO LORA images: torch.Size([118287, 1, 768])
# Shape of COCO LORA text embeddings: 118287
# Image IDs in COCO LORA dataset: [391895, 522418, 184613, 318219, 554625]
# Shape of COCO BASE images: torch.Size([118287, 1, 768])
# Shape of COCO BASE text embeddings: 118287
# Image IDs in COCO BASE dataset: [391895, 522418, 184613, 318219, 554625]