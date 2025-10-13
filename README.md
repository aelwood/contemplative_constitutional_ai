# Contemplative Constitutional AI

Constitutional AI finetuning using contemplative principles on existing open-source models. This project applies Anthropic's Constitutional AI framework to align pre-trained models (like QWEN) with contemplative wisdom traditions.

## Overview

This repository implements direct Constitutional AI finetuning on existing models to improve alignment with contemplative principles:

1. **Emptiness** - Understanding interdependence and avoiding conceptual rigidity
2. **Non-duality** - Recognizing unity while maintaining practical distinctions  
3. **Boundless Care** - Universal compassion and concern for all beings
4. **Mindfulness** - Present-moment awareness and clear discernment

## Approach

We start with pre-trained models (QWEN, Llama, Mistral) and apply the Constitutional AI process directly:

1. **Model Selection**: Choose base model (e.g., QWEN2.5-7B-Instruct)
2. **Constitutional Critique**: Model critiques its responses using contemplative principles
3. **Preference Learning**: Train model to prefer contemplatively-aligned responses
4. **Evaluation**: Test on alignment benchmarks and contemplative metrics

**No separate supervised learning phase needed** - we leverage the existing instruction-following capabilities of pre-trained models.

## Target Models

- **QWEN 2.5** (7B, 14B, 32B) - Primary focus
- Llama 3.1/3.2 models  
- Mistral models
- Other instruction-tuned models

## Quick Start

### Proof of Concept (MacBook M2)
```bash
# Install dependencies for Apple Silicon
pip install -r requirements.txt

# Generate preference pairs with your constitution
python scripts/generate_cai_data.py \
    --constitution contemplative-constitution-1.md \
    --prompts data/prompts/demo_prompts.jsonl \
    --model qwen2_0_5b \
    --output results/demo_preference_pairs.jsonl \
    --device mps

# Train a LoRA adapter with the generated data
python scripts/train_dpo.py \
    --dataset results/demo_preference_pairs.jsonl \
    --base-model qwen2_0_5b \
    --output models/qwen-0.5b-custom-constitution \
    --device mps

# Run contemplative CAI finetuning on QWEN2-0.5B (PoC)
python src/train_constitutional.py \
    --model Qwen/Qwen2-0.5B-Instruct \
    --principles contemplative \
    --device mps \
    --batch_size 1 \
    --max_pairs 500

# Evaluate on AILuminate demo dataset
python src/evaluate.py \
    --model ./models/qwen-0.5b-contemplative-poc \
    --benchmarks ailuminate_demo
```

### Development Scale (Cloud GPUs)
```bash
# Run contemplative CAI finetuning on QWEN2.5-7B
python src/train_constitutional.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --principles contemplative \
    --device cuda

# Evaluate on full alignment benchmarks
python src/evaluate.py \
    --model ./models/qwen-contemplative \
    --benchmarks alignment_full
```

## Methodology

### Constitutional AI Process
1. **Generate responses** from base model
2. **Critique using contemplative principles**
3. **Revise responses** based on critiques  
4. **Create preference pairs** (original vs revised)
5. **Train with DPO/PPO** on preference data

### Contemplative Constitution
- Responses should reflect interdependence (emptiness)
- Avoid reinforcing harmful dualistic thinking (non-duality)
- Show genuine care for all beings (boundless care)
- Encourage present-moment clarity (mindfulness)

## Evaluation

Compare finetuned models against base models on:
- **Alignment benchmarks**: HHH, TruthfulQA, etc.
- **Contemplative metrics**: Custom evaluations for contemplative principles
- **General capabilities**: Ensure no degradation in core abilities

## Repository Structure

```
├── src/
│   ├── train_constitutional.py    # Main training script
│   ├── evaluate.py               # Evaluation pipeline
│   └── utils/                    # Training utilities
├── data/
│   ├── constitutions/           # Contemplative principles
│   └── benchmarks/              # Evaluation datasets
├── configs/                     # Model and training configs
├── results/                     # Experimental results
├── DATA_PIPELINE.md             # Detailed data collection and processing
├── EVALUATION_METRICS.md        # Comprehensive evaluation methodology
├── IMPLEMENTATION_PLAN.md       # Phase-by-phase development plan
├── HARDWARE_REQUIREMENTS.md     # Hardware specs and configurations
└── DESIGN.md                    # Technical design document
```

## Based on Research

This implements the contemplative principles from "Contemplative Alignment" (arXiv:2504.15125), extending the prompting experiments to full constitutional finetuning.

Our implementation follows the proven Constitutional AI methodology from [Hugging Face's Constitutional AI with Open LLMs](https://huggingface.co/blog/constitutional_ai), adapting their scalable approach for contemplative principles.
