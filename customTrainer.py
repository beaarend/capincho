from transformers import Trainer
from decoder import hugging_decoder, OPT
from safetensors.torch import load_model, save_model
from textLoader import TextLoader
import torch
from transformers import TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
from datasets import Dataset


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        embeddings = inputs['embeddings']
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        captions = inputs['captions']
        if model.module.add_noise:
            embeddings = model.noise_injection(embeddings)

        prefix_tokens = model.module.mapper(embeddings).view(-1, model.prefix_length, model.hidden_size)
        captions_emb = model.module.get_input_embeds(captions).to(model.device, dtype=model.fp)
        # print(prefix_tokens.shape, embeddings.shape, captions_emb.shape)

        # [batch, bos + prefix + caption, d_model]
        input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(model.fp)

        labels = inputs['captions']
        # opt ignores -100 labels during loss computation
        ignore = torch.ones(labels.shape[0], model.prefix_length + 1).to(model.device) * -100

        labels = torch.concat([ignore, labels[:, 1:]], dim=1)
        out = model.model(inputs_embeds=input_emb, labels=labels.to(torch.long), ).loss
        # print(out)
        return out


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
    parser.add_argument('--prefix_len', type=int, default=10, help='model prefix length')
    parser.add_argument('--fp', choices=['fp16', 'fp32'], default='fp32', help='float precision')
    parser.add_argument('--output', type=str, default='opt350m-coco', help='output file basename no extension')
    parser.add_argument('--text_only', action='store_true', help='train using text only')
    parser.add_argument('--full_finetune', action='store_true', help='fine tune entire model', default=False)
    parser.add_argument('--noise', action='store_true', help='add noise to embeddings', default=False)
    parser.add_argument('--variance', type=float, help='variance for noise injection', default=0.016)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.float16 if args.fp == 'fp16' else torch.float32
    decoder = hugging_decoder(args.model_name, device, prefix_length=args.prefix_len, precision=precision, add_noise=args.noise,
                  variance=args.variance)

    if not args.full_finetune:
        decoder.lora_model(args.rank, args.alpha, args.dropout)
        print("Lora model")

    data = TextLoader(f'embeddings/{args.embeddings}', has_embeddings=True)
    tokens = decoder.tokenizer(data[:]['captions'], return_tensors="pt", padding=True).input_ids.squeeze(0)

    data_dict = {'embeddings': data[:]['embeddings'], 'captions': tokens}
    data = Dataset.from_dict(data_dict)
    optim = AdamW(decoder.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=args.epochs * len(data))
    training_args = TrainingArguments(
        output_dir="hugging",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_strategy="no",
        remove_unused_columns=False,
        do_eval=False,
    )
    trainer = CustomTrainer(model=decoder, train_dataset=data, optimizers=(optim, scheduler), args=training_args)
    trainer.train()
    save_model(trainer.model, args.output)

    # load_model(decoder, args.output)


