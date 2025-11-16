#!/usr/bin/env python3
"""
Web Interface for Math Word Problem Model with Visualization

Provides an interactive Gradio interface showing:
- Input problem
- Generated answer
- Token-by-token generation process
- Confidence scores

Usage:
    python web_inference.py
    python web_inference.py --model_path ./trained_model --port 7860
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import numpy as np


class ModelInference:
    """Handle model loading and inference with visualization."""
    
    def __init__(self, model_path="./trained_model"):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_with_trace(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
        """Generate answer with step-by-step trace."""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Track generation process
        generated_tokens = []
        token_probs = []
        generation_steps = []
        
        current_ids = input_ids
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get model predictions
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-p filtering
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Get probability of selected token
                token_prob = probs[0, next_token.item()].item()
                
                # Decode token
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                # Store information
                generated_tokens.append(token_text)
                token_probs.append(token_prob)
                generation_steps.append({
                    'step': step + 1,
                    'token': token_text,
                    'probability': token_prob
                })
                
                # Append to current sequence
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode full output
        full_output = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
        
        return full_output, generation_steps
    
    def format_generation_trace(self, steps):
        """Format generation trace as HTML."""
        html = "<div style='font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 5px;'>"
        html += "<h3 style='margin-top: 0;'>🔍 Token Generation Trace</h3>"
        html += "<table style='width: 100%; border-collapse: collapse;'>"
        html += "<tr style='background: #e0e0e0;'><th style='padding: 8px; text-align: left;'>Step</th><th style='padding: 8px; text-align: left;'>Token</th><th style='padding: 8px; text-align: left;'>Confidence</th></tr>"
        
        for step in steps:
            confidence_pct = step['probability'] * 100
            color = self._get_confidence_color(step['probability'])
            html += f"<tr style='border-bottom: 1px solid #ddd;'>"
            html += f"<td style='padding: 8px;'>{step['step']}</td>"
            html += f"<td style='padding: 8px; font-weight: bold;'>{step['token']}</td>"
            html += f"<td style='padding: 8px;'><span style='background: {color}; padding: 2px 8px; border-radius: 3px;'>{confidence_pct:.1f}%</span></td>"
            html += "</tr>"
        
        html += "</table></div>"
        return html
    
    def _get_confidence_color(self, prob):
        """Get color based on confidence level."""
        if prob > 0.7:
            return '#4caf50'  # Green
        elif prob > 0.4:
            return '#ff9800'  # Orange
        else:
            return '#f44336'  # Red


def create_interface(model_path="./trained_model"):
    """Create Gradio interface."""
    inference = ModelInference(model_path)
    
    def predict(prompt, max_tokens, temperature, top_p):
        """Generate prediction with visualization."""
        if not prompt.strip():
            return "Please enter a math problem.", ""
        
        try:
            full_output, generation_steps = inference.generate_with_trace(
                prompt,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                top_p=top_p
            )
            
            trace_html = inference.format_generation_trace(generation_steps)
            
            return full_output, trace_html
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    # Create interface
    with gr.Blocks(title="Math Problem Solver", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧮 Math Word Problem Solver")
        gr.Markdown("Enter a math word problem and see how the model solves it step-by-step!")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="📝 Math Problem",
                    placeholder="Example: John has 5 apples and gets 3 more from Mary.",
                    lines=3
                )
                
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Max New Tokens"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)"
                    )
                
                generate_btn = gr.Button("🚀 Solve Problem", variant="primary", size="lg")
                
                gr.Examples(
                    examples=[
                        ["John has 5 apples and gets 3 more from Mary."],
                        ["Sara had 20 candies and she gave 8 to Ahmed."],
                        ["There are 15 books, 10 pencils, and 5 erasers."],
                    ],
                    inputs=prompt_input
                )
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="📊 Complete Answer",
                    lines=5,
                    interactive=False
                )
                
                trace_html = gr.HTML(label="🔍 Generation Process")
        
        generate_btn.click(
            fn=predict,
            inputs=[prompt_input, max_tokens_slider, temperature_slider, top_p_slider],
            outputs=[output_text, trace_html]
        )
        
        gr.Markdown("""
        ### 💡 Tips:
        - **Temperature**: Lower values (0.1-0.5) make output more deterministic, higher values (0.8-2.0) more creative
        - **Top-p**: Controls diversity of token selection (0.9 is a good default)
        - **Max Tokens**: Maximum number of new tokens to generate
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Web interface for math problem inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./trained_model",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web interface on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    demo = create_interface(args.model_path)
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
