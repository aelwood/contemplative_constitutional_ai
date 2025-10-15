#!/usr/bin/env python3
"""
Compare base model vs fine-tuned model responses.
Tests the effect of constitutional DPO training.
"""

import sys
import json
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_base_model(model_key="qwen2_0_5b", device="mps"):
    """Load the base model."""
    print(f"Loading base model: {model_key}")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    
    return model, tokenizer


def load_finetuned_model(base_model, adapter_path, device="mps"):
    """Load the fine-tuned model with LoRA adapters."""
    print(f"Loading fine-tuned model from: {adapter_path}")
    
    # Load the LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device,
    )
    
    return model


def generate_response(model, tokenizer, prompt, max_new_tokens=200, device="mps"):
    """Generate a response from the model."""
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (not the prompt)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def load_training_prompts(dataset_path, max_prompts=5):
    """Load prompts from the training dataset."""
    prompts = []
    seen = set()
    
    with open(dataset_path, 'r') as f:
        for line in f:
            if len(prompts) >= max_prompts:
                break
            
            data = json.loads(line)
            prompt = data.get('prompt', '')
            
            # Only use unique prompts
            if prompt and prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)
    
    return prompts


def main():
    print("=" * 80)
    print("Contemplative Constitutional AI - Model Comparison")
    print("=" * 80)
    print()
    
    # Configuration
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset_path = Path("results/generated_preference_pairs.jsonl")
    adapter_path = Path("models/contemplative_dpo_test")
    
    if not adapter_path.exists():
        print(f"Error: Fine-tuned model not found at {adapter_path}")
        return
    
    # Load models
    print(f"Using device: {device}")
    print()
    
    base_model, tokenizer = load_base_model(device=device)
    finetuned_model = load_finetuned_model(base_model, adapter_path, device=device)
    
    # Load test prompts
    print(f"Loading prompts from: {dataset_path}")
    prompts = load_training_prompts(dataset_path, max_prompts=3)
    print(f"Loaded {len(prompts)} prompts")
    print()
    
    # Compare responses
    for i, prompt in enumerate(prompts, 1):
        print("=" * 80)
        print(f"PROMPT {i}:")
        print("-" * 80)
        print(prompt)
        print()
        
        # Generate base model response
        print("BASE MODEL RESPONSE:")
        print("-" * 80)
        base_response = generate_response(base_model, tokenizer, prompt, device=device)
        print(base_response)
        print()
        
        # Generate fine-tuned model response
        print("FINE-TUNED MODEL RESPONSE:")
        print("-" * 80)
        finetuned_response = generate_response(finetuned_model, tokenizer, prompt, device=device)
        print(finetuned_response)
        print()
    
    print("=" * 80)
    print("Comparison complete!")
    print()
    print("Look for differences such as:")
    print("  - More mindful, present-moment awareness")
    print("  - Non-absolute, context-sensitive language")
    print("  - Acknowledgment of uncertainty and assumptions")
    print("  - Compassionate, harm-reducing suggestions")
    print("  - Integration of multiple perspectives")
    print("=" * 80)


if __name__ == "__main__":
    main()

