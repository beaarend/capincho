import os.path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

# class EmbeddingDataset(Dataset):
#     def __init__(self, path, n_captions=5):
#         assert os.path.exists(path), '{} does not exist'.format(path)
#         self.text_embeddings = []
#         with open(path, 'rb') as f:
#             data = pickle.load(f)
#             # for i in data['texts_embeddings']:
#             #     self.text_embeddings.append(i[:n_captions])
#             #     # print(i[:n_captions])
#             for i in data['texts_embeddings']:
#                 # Ensure this is a tensor of shape [n_captions, embedding_dim]
#                 tensor_i = torch.tensor(i[:n_captions])
#                 self.text_embeddings.append(tensor_i)
#             self.images = data['image_embeddings']
#             self.image_id = data['image_id']
#             self.image_name = data['image_name']
#         # print(self.image_id)

class EmbeddingDataset(Dataset):
    def __init__(self, path, n_captions=5, flag=None):
        assert os.path.exists(path), f'{path} does not exist'

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Debugging print: Check shapes of text embeddings
        # for idx, i in enumerate(data['texts_embeddings']):
        #     print(f"Text embedding {idx} shape: {np.array(i[:n_captions]).shape}")

        # Convert everything to tensors correctly
        # self.text_embeddings = [torch.tensor(i[:n_captions]).float() for i in data['texts_embeddings']]  # List of [5, 768] tensors
        # self.images = torch.tensor(data['image_embeddings']).float()  # [B, 768]

        if(flag=='retrieval'):
            self.images = data['image_embeddings'].float()
        else:
            self.images = torch.stack(data['image_embeddings']).float()
        
        self.text_embeddings = [i[:n_captions].clone().detach().float() for i in data['texts_embeddings']]

        self.image_id = data['image_id']
        self.image_name = data['image_name']

        # Debugging print: Check shapes of images
        # print(f"Image embeddings shape: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        payload = {'image_id': self.image_id[index],
                   'image_name': self.image_name[index],
                   'image_embeddings': self.images[index],
                   'texts_embeddings': self.text_embeddings[index],}

        return payload

    def get_loader(self, shuffle=False, batch_size=400):
        indices = np.arange(len(self.images))
        if shuffle:
            np.random.shuffle(indices)
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)
        return loader, indices


if __name__ == '__main__':
    #dataset0 = EmbeddingDataset('embeddings/coco_openclip_val.pkl')
    dataset = EmbeddingDataset('embeddings/coco_train.pkl')
    #dataset2 = EmbeddingDataset('embeddings/coco_openclip_adapter_val.pkl')

    # print(len(dataset[:]['image_embeddings']), len(dataset[:]['texts_embeddings']), len(dataset[:]['captions']))
    # print(len(dataset2[:]['image_embeddings']), len(dataset2[:]['texts_embeddings']), len(dataset2[:]['captions']))
