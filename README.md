# Contemplative Constitutional AI

Constitutional AI finetuning using contemplative principles on existing open-source models. This project applies Anthropic's Constitutional AI framework to align pre-trained models (like QWEN) with contemplative wisdom traditions.

## Overview

This repository implements direct Constitutional AI finetuning on existing models to improve alignment with contemplative principles:

1. **Emptiness** - Understanding interdependence and avoiding conceptual rigidity
2. **Non-duality** - Recognizing unity while maintaining practical distinctions  
3. **Boundless Care** - Universal compassion and concern for all beings
4. **Mindfulness** - Present-moment awareness and clear discernment

### ✨ Key Features

- **AILuminate Integration**: 1,290 adversarial prompts from MLCommons benchmark (git submodule)
- **Train/Test Split Management**: Persistent, reproducible splits across all experiments
- **Flexible Filtering**: By hazard category (14 types) and persona type (3 types)
- **Multiple Model Support**: QWEN 2.5 (7B-32B), Llama, Mistral
- **Apple Silicon Optimized**: MPS acceleration for local development

## Approach

We start with pre-trained models (QWEN, Llama, Mistral) and apply the Constitutional AI process directly:

1. **Load Adversarial Prompts**: From AILuminate benchmark (designed to elicit unsafe responses)
2. **Generate Baseline Responses**: Model responds to adversarial prompts
3. **Constitutional Critique**: Model critiques its responses using contemplative principles
4. **Generate Revisions**: Model revises responses to align with principles
5. **Create Preference Pairs**: Original (rejected) vs. Revised (chosen)
6. **Train with DPO**: Direct Preference Optimization on preference pairs
7. **Evaluate**: Test on safety benchmarks and contemplative metrics

**No separate supervised learning phase needed** - we leverage the existing instruction-following capabilities of pre-trained models.

## Quick Start

### Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/contemplative_constitutional_ai.git
cd contemplative_constitutional_ai

# 2. Initialize AILuminate submodule
git submodule update --init --recursive

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python scripts/smoke_test.py
```

### Generate Preference Pairs with AILuminate

```bash
# Generate 100 preference pairs from AILuminate dataset with train/test split
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --max-prompts 100 \
    --hazard-categories vcr cse hte ssh \
    --create-split \
    --test-size 0.1 \
    --output results/ailuminate_pairs.jsonl \
    --device mps  # or cuda for GPUs

# This creates:
# - results/ailuminate_pairs.jsonl (400 preference pairs: 100 prompts × 4 principles)
# - data/splits/default_split.json (train/test split configuration)
```

### Train with DPO

```bash
# Train using the split configuration
python scripts/train_dpo.py \
    --dataset results/ailuminate_pairs.jsonl \
    --base-model qwen2_7b \
    --use-split-config \
    --output models/qwen-7b-contemplative \
    --epochs 3 \
    --device mps  # or cuda for GPUs

# The trainer automatically:
# - Loads the split configuration
# - Trains on training set
# - Evaluates on test set
```

### Advanced Usage

```bash
# Filter by specific hazard categories
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --hazard-categories vcr cse ssh \
    --persona-types skilled \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --create-split \
    --output results/physical_hazards.jsonl

# Generate only training split
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --split-only train \
    --split-config data/splits/default_split.json \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --output results/train_pairs.jsonl
```

## Target Models

- **QWEN 2.5** (0.5B, 1.5B, 7B, 14B, 32B) - Primary focus
  - 0.5B/1.5B for local PoC on MacBook M2
  - 7B+ for production quality
- Llama 3.1/3.2 models  
- Mistral models
- Other instruction-tuned models

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

## Dataset Capacity

**AILuminate Demo Dataset** (included as submodule):
- 1,290 prompts across 14 hazard categories
- 1,290 × 4 principles = **5,160 preference pairs**
- **Phase 0 (PoC)**: 500-1K pairs ✅ SUFFICIENT
- **Phase 1 (Dev)**: 5K-10K pairs ✅ SUFFICIENT

**AILuminate Practice Dataset** (requires MLCommons membership):
- 12,000 prompts
- 12,000 × 4 = 48,000 preference pairs  
- **Phase 2+ (Production)**: 40K+ pairs ✅ SUFFICIENT

**Anthropic HH-RLHF** (alternative/supplementary):
- 160,000 conversations
- See `IMPLEMENTATION_PLAN.md` for details

## Evaluation

Compare finetuned models against base models on:
- **Safety Benchmarks**: AILuminate (24K prompts, 14 hazard categories)
- **Contemplative Metrics**: Custom evaluations for 4 contemplative principles
- **Capability Benchmarks**: MT-Bench, MMLU, HumanEval
- **General Capabilities**: Ensure no degradation in core abilities

Evaluation using AILuminate includes:
- Harmfulness assessment (35% weight)
- Refusal clarity (25% weight)
- Bias mitigation (20% weight)
- Uncertainty acknowledgment (20% weight)

## Repository Structure

```
├── src/
│   ├── cai/
│   │   └── pipeline.py              # Constitutional AI pipeline
│   ├── constitutional/
│   │   └── config_parser.py         # Parse contemplative principles
│   ├── data/
│   │   ├── ailuminate_loader.py     # AILuminate dataset loader
│   │   └── split_manager.py         # Train/test split management
│   ├── models/
│   │   └── model_loader.py          # Model loading utilities
│   └── training/
│       └── dpo_trainer.py           # DPO training implementation
├── scripts/
│   ├── generate_cai_data.py         # Generate preference pairs
│   ├── train_dpo.py                 # Train with DPO
│   └── smoke_test.py                # Environment validation
├── data/
│   ├── constitutions/               # Contemplative principles
│   ├── benchmarks/
│   │   └── ailuminate/              # AILuminate submodule
│   └── splits/                      # Train/test split configs
├── configs/
│   ├── model_configs.yaml           # Model specifications
│   └── training_configs.yaml        # Training parameters
├── docs/
│   ├── AILUMINATE_INTEGRATION.md    # Integration details
│   └── AILUMINATE_USAGE.md          # Usage guide
├── results/                         # Experimental results
├── DATA_PIPELINE.md                 # Detailed data pipeline
├── EVALUATION_METRICS.md            # Evaluation methodology
├── IMPLEMENTATION_PLAN.md           # Phase-by-phase development
├── HARDWARE_REQUIREMENTS.md         # Hardware specifications
├── PROJECT_STATUS.md                # Current status and next steps
└── DESIGN.md                        # Technical design
```

## Documentation

- **[DATA_PIPELINE.md](DATA_PIPELINE.md)** - Data sources, AILuminate integration, train/test splits
- **[AILUMINATE_USAGE.md](docs/AILUMINATE_USAGE.md)** - Complete AILuminate usage guide
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Phase-by-phase development plan
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current status and priorities
- **[DESIGN.md](DESIGN.md)** - Technical architecture
- **[EVALUATION_METRICS.md](EVALUATION_METRICS.md)** - Evaluation framework
- **[HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md)** - Hardware specifications

## Based on Research

This implements the contemplative principles from "Contemplative Alignment" (arXiv:2504.15125), extending the prompting experiments to full constitutional finetuning.

Our implementation follows the proven Constitutional AI methodology from [Hugging Face's Constitutional AI with Open LLMs](https://huggingface.co/blog/constitutional_ai), adapting their scalable approach for contemplative principles.

**Dataset**: Uses [MLCommons AILuminate v1.0](https://github.com/mlcommons/ailuminate) benchmark for adversarial prompts and safety evaluation.
