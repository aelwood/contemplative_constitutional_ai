
# Design Document: Contemplative Constitutional AI

## Overview

This document outlines the technical design for implementing Constitutional AI finetuning with contemplative principles on existing pre-trained models like QWEN. The system uses simple markdown configuration files for constitutional phrases, integrates MLCommons AILuminate benchmark for comprehensive safety evaluation, and follows a phased implementation approach starting with a proof of concept using QWEN 2B before scaling to larger models.

## Architecture

### Core Components

1. **Base Model Loader** - Load and configure pre-trained models (QWEN 2B → 7B → larger models)
2. **Constitutional Config Parser** - Load contemplative principles from markdown files
3. **Data Pipeline Manager** - Integrate AILuminate, Anthropic HH, and custom datasets
4. **CAI Pipeline** - Generate critiques and revisions using constitutional principles
5. **Preference Data Creator** - Create training pairs from original/revised responses
6. **DPO Trainer** - Fine-tune using Direct Preference Optimization
7. **AILuminate Evaluator** - Primary benchmark for comprehensive safety evaluation
8. **Scaling Infrastructure** - llm-swarm for distributed generation and training

## Configuration System

### Constitutional Principles (Markdown Config)

All constitutional phrases are configured in `data/constitutions/contemplative_principles.md`:

```markdown
# Contemplative Constitutional Principles

## Emptiness Principle

### Critique Template
Does this response acknowledge the interdependent nature of phenomena and avoid treating concepts as inherently existing entities?

### Revision Guideline  
Revise the response to reflect how all things arise in dependence upon causes and conditions. Avoid presenting any phenomenon as existing independently or permanently.

### Example Application
- Acknowledge interconnectedness of issues
- Avoid absolute statements about identity or permanence
- Recognize multiple perspectives and contextual dependencies

## Non-duality Principle

### Critique Template
Does this response avoid reinforcing harmful us-vs-them thinking while maintaining practical distinctions where necessary?

### Revision Guideline
Acknowledge different perspectives while recognizing underlying unity. Avoid language that unnecessarily polarizes or creates artificial separations.

### Example Application
- Bridge opposing viewpoints when possible
- Use inclusive language
- Recognize shared humanity across differences

## Boundless Care Principle

### Critique Template
Does this response demonstrate genuine concern for the wellbeing of all beings, extending care beyond immediate stakeholders?

### Revision Guideline
Express care and consideration that includes all affected parties, including those not explicitly mentioned. Consider long-term and indirect impacts.

### Example Application
- Consider impact on marginalized groups
- Include environmental and future considerations
- Show empathy for all parties in conflicts

## Mindfulness Principle

### Critique Template
Does this response encourage present-moment awareness and clear discernment rather than reactive or habitual thinking?

### Revision Guideline
Focus on direct experience and clear observation rather than abstract speculation or automatic reactions. Encourage thoughtful reflection.

### Example Application
- Encourage pausing before reacting
- Focus on observable facts vs interpretations
- Promote self-awareness and reflection
```

## Technical Implementation

### 1. Model Selection Strategy

**Proof of Concept**: QWEN2-0.5B-Instruct (primary) and QWEN2-1.5B-Instruct (stretch)
- Optimized for Apple Silicon MacBook M2 (16GB unified memory)
- MPS acceleration for efficient local development
- Fast iteration and debugging without cloud costs
- Validate methodology before scaling to cloud GPUs

**Small Scale Development**: QWEN2.5-7B-Instruct  
- Good instruction following capabilities
- Strong reasoning for constitutional AI
- Efficient training requirements

**Production Scale**: QWEN2.5-14B/32B-Instruct
- Enhanced reasoning for complex contemplative scenarios
- Better generalization across diverse prompts
- Production-ready performance

**Alternative Targets**: 
- Llama-3.1/3.2 models (7B, 8B, 70B)
- Mistral models (7B, 22B)

### 2. Constitutional AI Pipeline

```python
def load_constitutional_config(config_path):
    """Load constitutional principles from markdown file"""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Parse markdown sections into structured principles
    principles = parse_markdown_principles(content)
    return principles

def apply_constitutional_ai(prompt, response, principles):
    """Apply CAI process using configured principles"""
    preference_pairs = []
    
    for principle in principles:
        # Generate critique
        critique_prompt = f"""
        {principle.critique_template}
        
        Human: {prompt}
        Assistant: {response}
        
        Critique:"""
        
        critique = model.generate(critique_prompt)
        
        # Generate revision
        revision_prompt = f"""
        Original: {response}
        Critique: {critique}
        
        {principle.revision_guideline}
        
        Revised response:"""
        
        revised = model.generate(revision_prompt)
        
        preference_pairs.append({
            'prompt': prompt,
            'chosen': revised,
            'rejected': response,
            'principle': principle.name
        })
    
    return preference_pairs
```

### 3. Training Approach

**Direct Preference Optimization (DPO)**:
- More stable than PPO
- No reward model needed
- Direct optimization on preference pairs

```python
# Training configuration (scalable across hardware)
training_configs = {
    'poc_macbook_m2': {
        'method': 'DPO',
        'learning_rate': 1e-6,
        'batch_size': 1,
        'gradient_accumulation_steps': 4,
        'epochs': 3,
        'beta': 0.1,
        'device': 'mps',  # Apple Metal Performance Shaders
        'fp16': True,
        'max_memory_mb': 12000
    },
    'development_gpu': {
        'method': 'DPO',
        'learning_rate': 1e-6,
        'batch_size': 4,
        'gradient_accumulation_steps': 8,
        'epochs': 3,
        'beta': 0.1,
        'device': 'cuda',
        'fp16': True
    }
}
```

## Evaluation Framework

### Primary Benchmark: MLCommons AILuminate

**AILuminate v1.0** serves as our primary evaluation benchmark for comprehensive AI risk assessment:
- 24,000 human-generated test prompts across 12 hazard categories
- Covers physical hazards (violence, self-harm), non-physical hazards (hate, defamation), and contextual hazards (specialized advice)
- Standardized evaluation methodology with tuned ensemble of safety evaluation models
- Available in multiple languages (English, French, with Chinese and Hindi coming)

### Evaluation Pipeline

```python
def evaluate_model(model_path, scale='small'):
    """Comprehensive evaluation pipeline with configurable scale"""
    results = {}
    
    # Primary benchmark - AILuminate
    if scale == 'poc':
        # Use demo dataset (1200 prompts) for proof of concept
        results['ailuminate'] = run_ailuminate_demo_eval(model_path)
    else:
        # Use full practice dataset (12K prompts) for development
        results['ailuminate'] = run_ailuminate_full_eval(model_path)
    
    # Capability preservation
    results['mmlu'] = evaluate_mmlu(model_path)
    results['hellaswag'] = evaluate_hellaswag(model_path)
    results['mt_bench'] = evaluate_mt_bench(model_path)
    
    # Custom contemplative evaluation
    results['contemplative'] = evaluate_contemplative_principles(model_path)
    
    # Safety robustness (jailbreak resistance)
    results['safety'] = evaluate_safety_robustness(model_path)
    
    return results
```

### Success Metrics

**Primary Success**: Improvement on ailuminate benchmark scores
**Secondary**: Maintained performance on capability benchmarks
**Qualitative**: Better responses on philosophical/ethical scenarios

## Implementation Plan

### Phase 0: Proof of Concept (Week 1-2)
**Goal**: Validate methodology with minimal computational requirements
- Repository structure and core infrastructure
- Constitutional config parser (markdown → structured principles)
- Basic model loading with QWEN2-0.5B/1.5B
- AILuminate demo dataset integration (1200 prompts)
- Simple CAI pipeline implementation
- Basic DPO training on 500-1000 preference pairs
- Evaluation with demo AILuminate + basic capability tests

### Phase 1: Small Scale Development (Week 3-4)
**Goal**: Full pipeline with QWEN2.5-7B and comprehensive evaluation
- Scale to QWEN2.5-7B-Instruct
- Full data pipeline integration (AILuminate practice + Anthropic HH)
- Enhanced CAI pipeline with all 4 contemplative principles
- DPO training on 5K-10K preference pairs
- Comprehensive evaluation suite implementation
- AILuminate practice dataset evaluation (12K prompts)

### Phase 2: Optimization and Scaling Infrastructure (Week 5-6)
**Goal**: Production-ready pipeline with distributed generation
- llm-swarm integration for distributed inference
- Large-scale dataset generation (40K preference pairs)
- Hyperparameter optimization and training efficiency
- Advanced evaluation metrics and monitoring
- Model checkpointing and versioning

### Phase 3: Production Scale Training (Week 7-8)
**Goal**: Train production models with full dataset
- QWEN2.5-14B/32B training with complete dataset
- Comprehensive safety and capability evaluation
- Comparative analysis across model sizes
- Documentation and reproducibility testing

### Phase 4: Validation and Analysis (Week 9-10)
**Goal**: Thorough evaluation and research analysis
- Full AILuminate benchmark evaluation
- Human expert evaluation of contemplative alignment
- Capability preservation analysis
- Research paper preparation and open-source release

## Repository Structure

```
contemplative_constitutional_ai/
├── README.md
├── DESIGN.md
├── data/
│   ├── constitutions/
│   │   └── contemplative_principles.md    # Constitutional config
│   ├── datasets/                          # Training data
│   └── evaluations/                       # Test cases
├── src/
│   ├── train_constitutional.py            # Main training script
│   ├── evaluate.py                        # Ailuminate evaluation
│   ├── config_parser.py                   # MD config parsing
│   └── utils/                             # Utilities
├── configs/
│   ├── model_configs.yaml                 # Model configurations
│   └── training_configs.yaml              # Training parameters
└── results/                               # Experimental outputs
```

## Computational Requirements

### Proof of Concept (QWEN2-0.5B/1.5B)
- **Hardware**: MacBook Pro M2 (16GB unified memory) with MPS acceleration
- **Alternative**: Single consumer GPU (RTX 4090, 24GB VRAM) on cloud
- **Training Time**: 3-6 hours on MacBook M2, 2-4 hours on GPU for 500-1000 preference pairs
- **Storage**: ~5GB for model and small dataset
- **Generation**: Local inference with Apple Silicon optimization, no cloud infrastructure needed

### Small Scale Development (QWEN2.5-7B)
- **Hardware**: 1-2 A100 GPUs (40GB VRAM) or equivalent
- **Training Time**: 12-24 hours for 5K-10K preference pairs
- **Storage**: ~25GB for datasets and models
- **Generation**: Single-node inference or small cluster

### Production Scale (QWEN2.5-14B/32B)
- **Hardware**: 4-8 A100 GPUs (40GB VRAM each) or H100 cluster
- **Training Time**: 48-96 hours for 40K preference pairs
- **Storage**: ~100GB for complete datasets and model checkpoints
- **Generation**: llm-swarm distributed inference on Slurm cluster

### Dataset Specifications by Phase
- **PoC**: 500-1K preference pairs (AILuminate demo subset)
- **Development**: 5K-10K pairs (AILuminate practice + selected HH)
- **Production**: 40K pairs (full integration of all data sources)
- **Evaluation**: AILuminate demo (1.2K) → practice (12K) → full benchmark