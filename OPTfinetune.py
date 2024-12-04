import argparse
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from util import model_size, learnable_parameters


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
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', )
    tokenizer = AutoTokenizer.from_pretrained(args.model, )
    model_size(model)
    learnable_parameters(model)

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
            resume_from_checkpoint=args.resume,
        )
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.save_dir)

