import argparse
import pickle
import time
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from captioningDataset import CaptioningDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from decoder import Decoder
from textLoader import TextLoader
from util import model_size, learnable_parameters


def train(epochs, batch_size, lr, filename, r, alpha, dropout, model_name, prefix_len, fp, text_only,
          full_finetune, schedule, add_noise, variance, save_history, dataset, root, dimension):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model, data, optimizer
    decoder = Decoder(model_name, device,
                      prefix_length=prefix_len,
                      precision=fp,
                      add_noise=add_noise,
                      variance=variance,
                      dimension=dimension)
    if not full_finetune:
        decoder.lora_model(r, alpha, dropout)
        print("Lora model")

    optim = AdamW(decoder.parameters(), lr=lr)

    model_size(decoder)
    learnable_parameters(decoder)

    if dataset == 'coco':
        train_data = CaptioningDataset(f'{filename}', text_only)
        val_name = filename.replace('train', 'val')
        val_data = CaptioningDataset(f'{val_name}', text_only)
    else:
        train_data = TextLoader(f'{filename}', has_embeddings=True, split='train')
        val_data = TextLoader(f'{filename}', has_embeddings=True, split='val')

    train_loader = train_data.get_loader(batch_size=batch_size)
    val_loader = val_data.get_loader(batch_size=batch_size)
    scheduler = None

    if schedule:
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=10,
                                                    num_training_steps=epochs * len(train_loader))

    save_path = os.path.join(args.save_path, 'checkpoint.pt')
    training_losses = []
    validation_losses = []

    # training loop
    for epoch in range(epochs):
        epoch_loss = []
        for batch in tqdm(train_loader):
            optim.zero_grad()
            output = decoder(batch)
            output.loss.backward()

            optim.step()
            if schedule:
                scheduler.step()
            loss = output.loss.detach().cpu().item()
            epoch_loss.append(loss)
            # print(f'allocated memory {(torch.cuda.memory_allocated(device="cuda:0") / 1e6)}')

        model_dict = {'epoch': epoch + 1,
                      'model_state_dict': decoder.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': epoch_loss[-1]
                      }
        training_losses.append(sum(epoch_loss) / len(epoch_loss))

        # validation
        epoch_val_losses = []
        for batch in val_loader:
            output = decoder(batch)
            epoch_val_losses.append(output.loss.detach().cpu().item())
        validation_losses.append(sum(epoch_val_losses) / len(epoch_val_losses))

        if save_history:
            path = save_path.split('.')[0]
            path += f'_epoch{epoch}.pt'
            torch.save(model_dict, path)
        else:
            torch.save(model_dict, save_path)

        print(f'saved model epoch {epoch + 1}')
        time.sleep(1)

        plt.plot(range(len(training_losses)), training_losses, label='training')
        plt.plot(range(len(validation_losses)), validation_losses, label='validation')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'training loss')
        plt.savefig(f'{root}/loss_plot.png')

        plt.clf()
        log = {'training_loss': training_losses, 'validation_loss': validation_losses}
        with open(f'{root}/loss_log.pkl', 'wb') as f:
            pickle.dump(log, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--embeddings', type=str, default='coco_openCLIP_train', help='embeddings filename')
    parser.add_argument('--rank', type=int, default=16, help='lora rank')
    parser.add_argument('--alpha', type=int, default=32, help='lora alpha parameter')
    parser.add_argument('--dropout', type=float, default=0.05, help='lora dropout parameter')
    parser.add_argument('--model_name', type=str, default="facebook/opt-350m", help='OPT model name')
    parser.add_argument('--prefix-len', type=int, default=10, help='model prefix length')
    parser.add_argument('--fp', choices=['fp16', 'fp32'], default='fp32', help='float precision')
    parser.add_argument('--text_only', action='store_true', help='train using text only')
    parser.add_argument('--full_finetune', action='store_true', help='fine tune entire model', default=False)
    parser.add_argument('--schedule', action='store_true', help='use linear scheduler', default=False)
    parser.add_argument('--noise', action='store_true', help='add noise to embeddings', default=False)
    parser.add_argument('--variance', type=float, help='variance for noise injection', default=0.016)
    parser.add_argument('--history', action='store_true', help='save epoch history', default=False)
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'geo', 'cxr'], help='dataset name')
    parser.add_argument('--save_path', default='/nethome/recpinfo/users/fibz/data/', help='root dir for saving results')
    parser.add_argument('--dimension', default=768, type=int, help='embedding dimension')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f'folders created: {args.save_path}')

    precision = torch.float16 if args.fp == 'fp16' else torch.float32
    train(args.epochs, args.batch_size, args.lr, args.embeddings, args.rank, args.alpha, args.dropout,
          args.model_name, args.prefix_len, precision, args.text_only, args.full_finetune, args.schedule,
          args.noise, args.variance, args.history, args.dataset, args.save_path, args.dimension)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.save_path, 'checkpoint.pt')
    with open(f'{args.save_path}/experiment.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
