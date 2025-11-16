#!/usr/bin/env python3
"""
Training script for fine-tuning language models on custom datasets.
Supports various models and customizable training parameters.
"""

import argparse
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import Dataset as HFDataset
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """Custom dataset for handling JSON training data."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data from JSON file
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats
        if isinstance(item, dict):
            if 'text' in item:
                text = item['text']
            elif 'prompt' in item and 'completion' in item:
                text = f"{item['prompt']}\n{item['completion']}"
            elif 'prompt' in item and 'operation' in item:
                # Handle math problem format with operation
                operation = item['operation']
                if operation['kind'] == 'add':
                    result = sum(operation['operands'])
                elif operation['kind'] == 'subtract':
                    result = operation['operands'][0] - operation['operands'][1]
                else:
                    result = 0
                text = f"{item['prompt']} The answer is {result}."
            elif 'instruction' in item:
                text = item['instruction']
                if 'input' in item and item['input']:
                    text += f"\n{item['input']}"
                if 'output' in item:
                    text += f"\n{item['output']}"
            else:
                text = str(item)
        else:
            text = str(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def load_and_prepare_data(train_file, test_file, tokenizer, max_length=512):
    """Load and prepare training and test datasets."""
    
    train_dataset = CustomDataset(train_file, tokenizer, max_length)
    test_dataset = CustomDataset(test_file, tokenizer, max_length) if test_file else None
    
    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune a language model on custom data'
    )
    
    # Data arguments
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='Path to training data file (JSON format)'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
        help='Path to test/validation data file (JSON format)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2',
        help='Model name or path (e.g., gpt2, gpt2-medium, distilgpt2)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./trained_model',
        help='Directory to save the trained model'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Training batch size per device'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=1000,
        help='Save checkpoint every X steps'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=100,
        help='Log every X steps'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='Evaluate every X steps'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use mixed precision training (FP16)'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Train file: {args.train_file}")
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)
    
    # Check device - support MPS for Apple Silicon
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and not args.no_cuda:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    
    # Load and prepare datasets
    logger.info("Loading datasets...")
    train_dataset, test_dataset = load_and_prepare_data(
        args.train_file,
        args.test_file,
        tokenizer,
        args.max_length
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if test_dataset else None,
        eval_strategy='steps' if test_dataset else 'no',
        save_total_limit=3,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_dir=f'{args.output_dir}/logs',
        seed=args.seed,
        load_best_model_at_end=True if test_dataset else False,
        metric_for_best_model='eval_loss' if test_dataset else None,
        report_to=['tensorboard'],
        save_strategy='steps',
        remove_unused_columns=False,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked LM
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate if test dataset is provided
    if test_dataset:
        logger.info("Evaluating on test set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
