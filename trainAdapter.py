import argparse
import json
import os.path
import time
import torch
from adapters import ContrastiveResidualAdapter, SigAdapter
from tqdm import tqdm
from torch.optim import Adam
import foundation_models
from embeddingsDataset import COCODataset
from earlyStopping import EarlyStopping
from util import plot_curves
device = torch.device("cuda" if torch.cuda.is_available() else "")


def run_training(identifier, batch_size, dataset, model, epochs, lr, patience, delta, restore_best=False):
    train_dataset = COCODataset(f'embeddings/{dataset}_train.pkl')
    val_dataset = COCODataset(f'embeddings/{dataset}_val.pkl')
    train_loader, train_indices = train_dataset.get_loader(shuffle=False, batch_size=batch_size)
    val_loader, val_indices = val_dataset.get_loader(shuffle=False, batch_size=batch_size)
    save_option = "best" if restore_best else "last"
    if patience < 0:
        patience = epochs

    es = EarlyStopping(patience=patience, minimal_improvement=delta, objective='minimize', save_option=save_option)
    training_losses = []
    validation_losses = []

    optim = Adam(model.parameters(), lr=lr)

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
    plot_curves(training_losses, validation_losses, f'{identifier}.png',
                'contrastive adapter')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openclip', choices=['openclip', 'clip', 'coca'],
                        help='foundation model')
    parser.add_argument('--adapter', type=str, default='contrastive', choices=['contrastive', 'sig'],
                        help='adapter type')
    parser.add_argument('--alpha', type=float, default=0.3, help='residual learning rate')
    parser.add_argument('--bias', type=float, default=-10., help='logit bias, sig adapter')
    parser.add_argument('--embeddings', type=str, default='coco_openclip',
                        help='embeddings path root, val and train will be appended later')
    parser.add_argument('--use_bias', action='store_true', help='use logit bias in sig adapter', default=True)
    parser.add_argument('--multiple_positives', action='store_true',
                        help='use multiple positives per batch in sig adapter', default=False)
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--embedding_dim', type=int, default=768, help='embedding dimension')
    parser.add_argument('--learnable_alpha', action='store_true', help='learnable alpha', default=False)
    parser.add_argument('--output', type=str, default='coco_openclip_contrastive', help='experiment name no extension')
    parser.add_argument('--patience', type=int, default=-1, help='early stopping patience, '
                                                                 'negative value means no early stopping')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--best', action='store_true', help='restore best model if using early stopping', default=False)
    parser.add_argument('--delta', type=float, help='minimal improvement for early stopping', default=0.01,)
    parser.add_argument('--epochs', type=int, default=200, help='number training of epochs')
    args = parser.parse_args()

    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP}
    foundation = model_dict[args.model](device)
    foundation.load_model()

    logit_scale = foundation.backbone.logit_scale
    bias = torch.ones([]) * args.bias

    if args.adapter == 'sig':
        model = SigAdapter(args.embedding_dim, args.alpha, bias, logit_scale, use_logit_bias=args.use_bias,
                           multi_positive=args.multiple_positives,)

    else:
        model = ContrastiveResidualAdapter(args.embedding_dim, args.alpha, logit_scale, args.learnable_alpha, )

    model.to(device)
    run_training(args.output, args.batch_size, args.embeddings, model, args.epochs, args.lr, args.patience,
                 args.delta, args.best)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = f'checkpoints/contrastive/{args.output}.pt'
    result_dict['logit_scale'] = model.logit_scale.detach().cpu().item()
    with open(f'experiments/{args.output}.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
