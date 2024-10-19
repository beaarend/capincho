import argparse
import time
import torch
import torch.nn as nn
from torch import dtype
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model
from captioningDataset import CaptioningDataset
from mapping import Mapper
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import json


class OPT(nn.Module):
    def __init__(self, model_name, device, precision=torch.float16, prefix_length=10, add_noise=True):
        super(OPT, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=precision,
            device_map='auto',

        )
        self.add_noise = add_noise
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.device = device
        self.hidden_size = self._get_hidden_size()
        self.prefix_length = prefix_length
        self.fp = precision
        self.mapper = Mapper(768, self.hidden_size, self.prefix_length).to(self.device, dtype=precision)

    def generate(self, prompt, stochastic=False, max_tokens=50, seed=32):
        if stochastic:
            set_seed(seed)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, do_sample=stochastic, max_new_tokens=max_tokens, )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    def caption(self, embeddings, stochastic=False, max_tokens=50, seed=32):
        if stochastic:
            set_seed(seed)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        bos = self.model.get_input_embeddings()(torch.tensor([2]).to(self.device, dtype=torch.long)).unsqueeze(0)
        prefix = self.mapper(embeddings.to(dtype=self.fp, device=self.device)).view(-1, self.prefix_length, self.hidden_size)
        prefix = torch.concat([bos, prefix], dim=1)
        generated_ids = self.model.generate(do_sample=stochastic, max_new_tokens=max_tokens, inputs_embeds=prefix)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def forward(self, batch):
        embeddings = batch['embeddings'].to(self.device, dtype=self.fp)
        captions = batch['captions']
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        if self.add_noise:
            embeddings = self.noise_injection(embeddings)

        prefix_tokens = self.mapper(embeddings).view(-1, self.prefix_length, self.hidden_size)
        captions_emb = self.get_input_embeds(captions).to(self.device, dtype=self.fp)

        # [batch, bos + prefix + caption, d_model]
        input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(self.fp)
        labels = self.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(
            self.device, self.fp)

        # opt ignores -100 labels during loss computation
        # print(input_emb.shape)
        labels = torch.concat([torch.ones(labels.shape[0], self.prefix_length + 1).to(self.device) * -100,
                               labels[:, 1:]], dim=1)
        # print(labels.shape)
        return self.model(inputs_embeds=input_emb, labels=labels.to(torch.long))

    def get_input_embeds(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(self.device).squeeze(0)
        embeddings = self.model.get_input_embeddings()
        return embeddings(input_ids)

    def _get_hidden_size(self):
        ids = self.tokenizer("prompt", return_tensors="pt").input_ids.to(self.device).squeeze(0)
        return self.model.get_input_embeddings()(ids).shape[1]

    def noise_injection(self, x, variance=0.016, ):
        return x + torch.randn(x.shape, device=self.device, dtype=self.fp) * math.sqrt(variance)

    def lora_model(self, r, alpha, dropout):
        for param in self.model.parameters():
            param.requires_grad = False
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",

        )
        self.model = get_peft_model(self.model, config).to(self.fp)


def train(epochs, batch_size, lr, filename, r, alpha, dropout, model_name, prefix_len, fp, output_name, text_only,
          full_finetune):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = OPT(model_name, device, prefix_length=prefix_len, precision=fp, add_noise=text_only)
    if not full_finetune:
        decoder.lora_model(r, alpha, dropout)

    dataset = CaptioningDataset(f'embeddings/{filename}.pkl', text_only)
    optim = AdamW(decoder.parameters(), lr=lr)
    loader = dataset.get_loader(shuffle=False, batch_size=batch_size)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=200, num_training_steps=epochs * len(loader))
    save_path = f'checkpoints/caption/{output_name}.pt'

    losses = []
    for epoch in range(epochs):
        for batch in tqdm(loader):
            optim.zero_grad()
            output = decoder(batch)
            output.loss.backward()
            optim.step()
            scheduler.step()
            loss = output.loss.detach().cpu().item()
            losses.append(loss)
            if math.isnan(loss):
                print('Loss is NaN')

            plt.plot(range(len(losses)), losses, label='loss')
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.title(f'training {output_name}')
            plt.savefig(f'plots/caption/{output_name}.png')
            plt.clf()

        model_dict = {'epoch': epoch + 1,
                      'model_state_dict': decoder.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': losses[-1]
                      }
        torch.save(model_dict, save_path)
        print(f'saved model epoch {epoch + 1}')
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--filename', type=str, default='coco_openCLIP_train', help='embeddings filename no extension')
    parser.add_argument('--rank', type=int, default=16, help='lora rank')
    parser.add_argument('--alpha', type=int, default=32, help='lora alpha parameter')
    parser.add_argument('--dropout', type=float, default=0.05, help='lora dropout parameter')
    parser.add_argument('--model_name', type=str, default="facebook/opt-350m", help='OPT model name')
    parser.add_argument('--prefix-len', type=int, default=10, help='model prefix length')
    parser.add_argument('--fp', choices=['fp16', 'fp32'], default='fp32', help='float precision')
    parser.add_argument('--output', type=str, default='opt1.3b-coco', help='output file basename without extension')
    parser.add_argument('--text_only', action='store_true', help='train using text only')
    parser.add_argument('--full_finetune', action='store_true', help='fine tune entire model')
    args = parser.parse_args()

    precision = torch.float16 if args.fp == 'fp16' else torch.float32
    train(args.epochs, args.batch_size, args.lr, args.filename, args.rank, args.alpha, args.dropout,
          args.model_name, args.prefix_len, precision, args.output, args.text_only, args.full_finetune)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = f'checkpoints/caption/{args.output}.pt'
    with open(f'experiments/{args.output}.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
