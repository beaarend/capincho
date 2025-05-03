import torch
import pickle
from embeddingsDataset import EmbeddingDataset

embed_lora = EmbeddingDataset('embeddings/rsicd_lora_train.pkl')
embed_base = EmbeddingDataset('embeddings/rsicd_nolora_train.pkl')

# Load embeddings and convert them to tensors
if isinstance(embed_lora.images, torch.Tensor):
    images_lora = embed_lora.images  # Already a tensor
else:
    images_lora = torch.stack(embed_lora.images)  # Stack if it's a list of tensors

if isinstance(embed_base.images, torch.Tensor):
    images_base = embed_base.images  # Already a tensor
else:
    images_base = torch.stack(embed_base.images)  # Stack if it's a list of tensors

# Normalize embeddings
images_lora = images_lora / images_lora.norm(dim=-1, keepdim=True)
images_base = images_base / images_base.norm(dim=-1, keepdim=True)

# Calculate cosine similarity in batches
batch_size = 64  # Adjust this batch size depending on your available memory
num_batches = (images_lora.size(0) + batch_size - 1) // batch_size  # To handle the last smaller batch

cos_sim_sum = 0.0  # To accumulate cosine similarities
num_elements = 0  # To accumulate the number of elements for averaging

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, images_lora.size(0))  # Ensure last batch is handled properly

    batch_images_lora = images_lora[start_idx:end_idx]
    batch_images_base = images_base[start_idx:end_idx]

    # Calculate cosine similarity for the batch
    cos_sim_batch = torch.nn.functional.cosine_similarity(batch_images_lora, batch_images_base, dim=-1)

    # Update the sum of cosine similarities and count of elements
    cos_sim_sum += cos_sim_batch.sum().item()
    num_elements += cos_sim_batch.size(0)

# Compute and print the average cosine similarity
mean_cos_sim = cos_sim_sum / num_elements
print(f"Mean cosine similarity between LORA and BASE image embeddings: {mean_cos_sim:.6f}")
