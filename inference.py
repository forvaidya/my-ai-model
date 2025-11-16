#!/usr/bin/env python3
"""
CLI Inference Script for Math Word Problem Model

Usage:
    python inference.py --prompt "John has 5 apples and he gets 3 more from Mary."
    python inference.py --interactive
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


def load_model(model_path="./trained_model"):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return tokenizer, model


def generate_answer(prompt, tokenizer, model, max_length=128, temperature=0.7, top_p=0.9):
    """Generate answer for a given math problem."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    return generated_text


def interactive_mode(tokenizer, model):
    """Run in interactive mode."""
    print("\n" + "="*70)
    print("Interactive Math Problem Solver")
    print("="*70)
    print("Enter a math word problem and I'll solve it!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            prompt = input("\n📝 Enter problem: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not prompt:
                print("Please enter a valid problem.")
                continue
            
            print("\n🤔 Thinking...")
            answer = generate_answer(prompt, tokenizer, model)
            
            print("\n" + "-"*70)
            print("📊 Result:")
            print("-"*70)
            print(f"{answer}")
            print("-"*70)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for math word problem model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./trained_model",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Math problem to solve"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    
    args = parser.parse_args()
    
    # Load model
    tokenizer, model = load_model(args.model_path)
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(tokenizer, model)
    elif args.prompt:
        print(f"\n📝 Problem: {args.prompt}")
        print("\n🤔 Thinking...")
        answer = generate_answer(
            args.prompt, 
            tokenizer, 
            model,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print("\n" + "="*70)
        print("📊 Result:")
        print("="*70)
        print(f"{answer}")
        print("="*70 + "\n")
    else:
        print("Error: Please provide --prompt or use --interactive mode")
        print("Examples:")
        print('  python inference.py --prompt "John has 5 apples and gets 3 more."')
        print("  python inference.py --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
