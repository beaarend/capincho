import argparse
import pickle
import time
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from captioningDataset import CaptioningDataset
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import json
from decoder import OPT
from textLoader import TextLoader
from torch import nn


def train(epochs, batch_size, lr, filename, r, alpha, dropout, model_name, prefix_len, fp, output_name, text_only,
          full_finetune, schedule, add_noise, variance, save_history, datset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decoder = OPT(model_name, device, prefix_length=prefix_len, precision=fp, add_noise=add_noise, variance=variance)

    if torch.cuda.device_count() > 1:
        decoder = nn.DataParallel(decoder)

    if not full_finetune:
        decoder.lora_model(r, alpha, dropout)
        print("Lora model")

    if datset == 'coco':
        dataset = CaptioningDataset(f'embeddings/{filename}', text_only)
    else:
        dataset = TextLoader(f'embeddings/{filename}', has_embeddings=True)

    optim = AdamW(decoder.parameters(), lr=lr)
    loader = dataset.get_loader(batch_size=batch_size)
    if schedule:
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=200, num_training_steps=epochs * len(loader))
    save_path = f'checkpoints/caption/{output_name}.pt'
    avg_losses = []
    for epoch in range(epochs):
        epoch_loss = []
        for batch in tqdm(loader):
            optim.zero_grad()
            output = decoder(batch)
            output.loss.backward()
            optim.step()
            if schedule:
                scheduler.step()
            loss = output.loss.detach().cpu().item()
            epoch_loss.append(loss)
            if math.isnan(loss):
                print('Loss is NaN')

        model_dict = {'epoch': epoch + 1,
                      'model_state_dict': decoder.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': epoch_loss[-1]
                      }
        avg_losses.append(sum(epoch_loss) / len(epoch_loss))

        if save_history:
            path = save_path.split('.')[0]
            path += f'_epoch{epoch}.pt'
            torch.save(model_dict, path)
        else:
            torch.save(model_dict, save_path)

        print(f'saved model epoch {epoch + 1}')
        time.sleep(1)

    plt.plot(range(len(avg_losses)), avg_losses, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'training {output_name}')
    plt.savefig(f'plots/experiment training/{output_name}.png')

    plt.clf()
    log = {'training_loss': avg_losses, 'validation_loss': []}
    with open(f'loss/{output_name}.pkl', 'wb') as f:
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
    parser.add_argument('--output', type=str, default='opt350m-coco', help='output file basename no extension')
    parser.add_argument('--text_only', action='store_true', help='train using text only')
    parser.add_argument('--full_finetune', action='store_true', help='fine tune entire model', default=False)
    parser.add_argument('--schedule', action='store_true', help='use linear scheduler', default=False)
    parser.add_argument('--noise', action='store_true', help='add noise to embeddings', default=False)
    parser.add_argument('--variance', type=float, help='variance for noise injection', default=0.016)
    parser.add_argument('--history', action='store_true', help='save epoch history', default=False)
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'geo', 'cxr'], help='dataset name')
    args = parser.parse_args()

    precision = torch.float16 if args.fp == 'fp16' else torch.float32
    train(args.epochs, args.batch_size, args.lr, args.embeddings, args.rank, args.alpha, args.dropout,
          args.model_name, args.prefix_len, precision, args.output, args.text_only, args.full_finetune, args.schedule,
          args.noise, args.variance, args.history, args.dataset)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = f'checkpoints/caption/{args.output}.pt'
    with open(f'experiments/{args.output}.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
