import clip
import numpy as np
import torch
from embeddingsDataset import EmbeddingDataset
from tqdm import tqdm
from adapters import ContrastiveResidualAdapter, SigAdapter
import matplotlib.pyplot as plt
import clip
import pickle
try:
    import open_clip
except ImportError:
    print('open_clip is not available')

device = torch.device("cuda:0" if torch.cuda.is_available() else "")


def evaluate_image_text(path, temperature, n=5, mode='one'):
    # dataset = EmbeddingDataset(path, n_captions=n, flag='retrieval')
    dataset = EmbeddingDataset(path, n_captions=n)
    loader, indices = dataset.get_loader(batch_size=5000, shuffle=False)
    result = []
    
    for batch in loader:
        images = batch['image_embeddings'].to(device).squeeze()
        captions = batch['texts_embeddings'].to(device).flatten(start_dim=0, end_dim=1)
        images = images / images.norm(dim=-1, keepdim=True)
        captions = captions / captions.norm(dim=-1, keepdim=True)
        sim = (images @ captions.T) * temperature.exp()
        # print(sim.shape)
        rank = sim.argsort(descending=True, dim=1) // 5
        for i in range(images.shape[0]):
            matches = (rank[i, :] == i)
            result.append(matches.nonzero().squeeze())

    results = torch.stack(result, dim=0)
    if mode == 'one':
        # check only the first caption, 1 correct caption
        results = results.squeeze()
        r1 = (results[:, 0] < 1).nonzero().shape[0] / results.shape[0]
        r5 = (results[:, 0] < 5).nonzero().shape[0] / results.shape[0]
        r10 = (results[:, 0] < 10).nonzero().shape[0] / results.shape[0]
        return {'R@1': r1, 'R@5': r5, 'R@10': r10}

    elif mode == 'all':
        # check all captions, 5 correct captions
        r5 = (results < 5).nonzero().shape[0] / (results.shape[0] * results.shape[1])
        r10 = (results < 10).nonzero().shape[0] / (results.shape[0] * results.shape[1])
        r15 = (results < 15).nonzero().shape[0] / (results.shape[0] * results.shape[1])
        return {'R@5': r5, 'R@10': r10, 'R@15': r15}

if __name__ == '__main__':
    model, _ = clip.load('ViT-L/14')
    mode = 'one'
    title = f'RSICD Retrieval with LORA Adapter (ViT-L/14)'
    path = 'embeddings/rsicd_lora_train_3.pkl'
    label = 'RSICD (LORA)'

    results = {}
    for t in [0.03, 0.07, 0.1, 0.15]:
        result = evaluate_image_text(path, torch.tensor(t), mode='all')
        results[f"T={t}"] = result

    # Print out the results nicely
    print(f"{label}:")
    for t, result in results.items():
        print(f"{t}:")
        for k, v in result.items():
            print(f"  {k} = {v:.4f}")

    # Plot the results
    for t, result in results.items():
        k_ints = [int(k.split('@')[1]) for k in result.keys()]
        values = list(result.values())
        plt.plot(k_ints, values, '-o', label=f"{label} (T={t})")

    plt.legend()
    plt.xlabel('k')
    plt.ylabel('R@k')
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.show()