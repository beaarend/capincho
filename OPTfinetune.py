import argparse
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from util import model_size, learnable_parameters
import os
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='facebook/opt-350m')
    parser.add_argument('--dataset', type=str, default='textDatasets/publico-COMPLETO.txt')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints/opt-finetune')
    parser.add_argument('--fp16', action='store_true', default=False, help='use 16-bits floating point precision')
    parser.add_argument('--resume', default=False, help='resume from checkpoint', action="store_true")
    parser.add_argument('--lora', action='store_true', default=False, help='Low Rank Adaptation')
    parser.add_argument('--rank', type=int, default=16, help='rank for Low Rank Adaptation')
    parser.add_argument('--alpha', type=float, default=32, help='alpha for Low Rank Adaptation')
    parser.add_argument('--accumulate_grad_steps', type=int, default=1, help='number of steps to accumulate grad')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size per device')
    args = parser.parse_args()

    if args.resume:
        assert os.path.exists(args.output_dir), 'output directory does not exist'
        checkpoints = glob.glob(f'{args.output_dir}/checkpoint-*')
        assert len(checkpoints) > 0, f'no checkpoints found at {args.output_dir}'
        steps = [c.split('-')[-1] for c in checkpoints]
        steps.sort(reverse=True)
        model = AutoModelForCausalLM.from_pretrained(f'{args.output_dir}/checkpoint-{steps[0]}')
        print(f'loaded checkpoint-{steps[0]}')

    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', )
        if args.lora:
            for param in model.parameters():
                param.requires_grad = False
            config = LoraConfig(
                r=args.rank,
                lora_alpha=args.alpha,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.add_adapter(config, adapter_name='adapter-pt')

    tokenizer = AutoTokenizer.from_pretrained(args.model, )
    model_size(model)
    learnable_parameters(model)

    data = load_dataset('text', data_files=args.dataset, encoding='utf8', cache_dir=args.output_dir)
    data = data.map(lambda sample: tokenizer(sample['text']), batched=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),

        args=transformers.TrainingArguments(
            fp16=args.fp16,
            logging_steps=500,
            logging_strategy='steps',
            learning_rate=args.lr,
            output_dir=args.output_dir,
            save_strategy='steps',
            save_steps=500,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulate_grad_steps,
            num_train_epochs=args.epochs,
            overwrite_output_dir=True,

        )
    )
    trainer.train()
    trainer.save_model(args.save_dir)

