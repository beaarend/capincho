import argparse
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='facebook/opt-350m')
    parser.add_argument('--tokenizer', type=str, default='facebook/opt-350m')
    parser.add_argument('--dataset', type=str, default='textDatasets/publico-COMPLETO.txt')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints/opt-finetune')
    parser.add_argument('--fp16', action='store_true', default=False, help='use 16-bits floating point precision')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', )
    tokenizer = AutoTokenizer.from_pretrained(args.model, )

    data = load_dataset('text', data_files=args.dataset, encoding='utf8', cache_dir=args.output_dir)
    data = data.map(lambda sample: tokenizer(sample['text']), batched=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),

        args=transformers.TrainingArguments(
            fp16=args.fp16,
            logging_steps=200,
            logging_strategy='steps',
            learning_rate=args.lr,
            output_dir=args.output_dir,
            save_strategy='steps',
            save_steps=50,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            overwrite_output_dir=True,

        )
    )
    trainer.train()
    trainer.save_model(args.save_dir)

