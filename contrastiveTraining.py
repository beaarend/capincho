import os
import pickle
import time
import torch
from adapters import ContrastiveResidualAdapter, DynamicContrastiveResidualAdapter, SigAdapter
from tqdm import tqdm
from torch.optim import Adam
import foundation_models
from util import plot_curves
from geoVQALoader import GeoVQADataset
from embeddingsLoader import COCODataset
from earlyStopping import EarlyStopping
device = torch.device("cuda" if torch.cuda.is_available() else "")


def adapt_features(model,
                   checkpoint_path='checkpoints/contrastive/clip_residual_MPT.pt',
                   dataset_path='datasets_torchvision/embeddings/coco_ViTL_val.pkl',
                   save_path='datasets_torchvision/embeddings/coco_MPT.pkl',):

    assert os.path.exists(checkpoint_path), f'No checkpoint found at {checkpoint_path}'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    dataset = COCODataset(dataset_path)
    loader, indices = dataset.get_loader(batch_size=len(dataset), shuffle=False)

    for batch in loader:
        images = model.image_projection(batch[0]).cpu()
        texts = model.text_projection(batch[1]).cpu()
        results = {'image_features': images, 'text_features': texts}

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    print('done!')


def run_training(identifier, batch_size, dataset, model, initial_alpha, initial_bias, epochs):
    train_dataset = COCODataset(f'embeddings/{dataset}_train.pkl')
    val_dataset = COCODataset(f'embeddings/{dataset}_val.pkl')
    train_loader, train_indices = train_dataset.get_loader(shuffle=False, batch_size=batch_size)
    val_loader, val_indices = val_dataset.get_loader(shuffle=False, batch_size=batch_size)
    es = EarlyStopping(patience=200, minimal_improvement=0.01, objective='minimize', save_option='last')
    training_losses = []
    validation_losses = []

    optim = Adam(model.parameters(), lr=0.00001)

    print(f'training {identifier}')
    time.sleep(1)

    for i in tqdm(range(epochs)):
        training_loss = model.train_epoch(train_loader, optim)
        validation_loss = model.val_epoch(val_loader)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        model_dict = {'epoch': i,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': training_losses[-1]
                      }
        es.update(validation_loss, model_dict)
        if es.stop:
            break

    torch.save(es.model_to_save(), f'checkpoints/contrastive/{identifier}.pt')
    plot_curves(training_losses, validation_losses, identifier, 'loss')


if __name__ == '__main__':
    foundation = foundation_models.OpenCLIP(device)
    foundation.load_model()
    logit_scale = foundation.backbone.logit_scale
    bias = torch.ones([]) * -10.0
    for batch_size in [400]:
        for alpha in [0.3, ]:
            # model = SigAdapter(768, alpha, torch.tensor(-10.0), logit_scale, use_logit_bias=True, multi_positive=True,)

            model = ContrastiveResidualAdapter(768, alpha, logit_scale, False, ).to(device)
            model.to(device)
            # run_training(f'COCA_residual_adapter_{alpha}', batch_size, 'coco_COCA', model,
            #              alpha, bias, 200)
            adapt_features(model,
                           checkpoint_path=f'checkpoints/contrastive/OpenCLIP_residual_adapter_0.3.pt',
                           save_path=f'embeddings/coco_contrastive_train.pkl',
                           dataset_path=f'embeddings/coco_openCLIP_train.pkl')

