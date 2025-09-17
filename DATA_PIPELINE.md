# Data Pipeline Design

## Overview

This document outlines the data collection, processing, and management pipeline for the Contemplative Constitutional AI project. Our approach follows the proven methodology from [Hugging Face's Constitutional AI implementation](https://huggingface.co/blog/constitutional_ai) while adapting it for contemplative principles.

## Data Sources

### Primary Dataset Sources

1. **MLCommons AILuminate Benchmark**
   - Source: `https://github.com/mlcommons/ailuminate`
   - Purpose: Comprehensive AI risk assessment with human-generated hazardous prompts
   - Size: 24K prompts (12K public practice + 12K private + 1.2K demo)
   - Hazard categories: 12 categories including violent crimes, hate, privacy violations, specialized advice
   - Usage: High-quality adversarial prompts for constitutional AI training
   - Languages: English (primary), French, with Simplified Chinese and Hindi coming in 2025

2. **Anthropic HH-RLHF Dataset**
   - Source: `https://huggingface.co/datasets/Anthropic/hh-rlhf`
   - Purpose: Red-teaming prompts that elicit potentially problematic responses
   - Size: ~160K conversations
   - Usage: Extract prompts for constitutional AI process

3. **OpenAssistant Conversations**
   - Source: `https://huggingface.co/datasets/OpenAssistant/oasst1`
   - Purpose: High-quality human conversations for diverse prompt generation
   - Size: ~66K conversation trees
   - Usage: Philosophical and ethical discussion prompts

4. **Custom Contemplative Scenarios**
   - Source: Generated internally
   - Purpose: Specific scenarios testing contemplative principles
   - Target size: ~5K prompts
   - Focus areas:
     - Ethical dilemmas requiring interdependence thinking
     - Conflict resolution scenarios testing non-duality
     - Situations requiring boundless care
     - Decision-making scenarios testing mindfulness

### Prompt Distribution Strategy

Target distribution for constitutional AI dataset generation:

```yaml
prompt_categories:
  ailuminate_hazards: 40%    # AILuminate benchmark prompts across 12 hazard categories
  ethical_dilemmas: 25%      # Complex moral decisions
  philosophical_questions: 20%  # Questions about nature of reality, consciousness
  practical_advice: 15%     # Daily life guidance with ethical implications

ailuminate_hazard_distribution:
  # Physical Hazards
  violent_crimes: 8%         # vcr
  sex_related_crimes: 6%     # src  
  child_exploitation: 4%     # cse
  suicide_self_harm: 6%      # ssh
  weapons_cbrne: 4%          # iwp
  
  # Non-Physical Hazards  
  ip_violations: 8%          # ipv
  defamation: 6%             # dfm
  non_violent_crimes: 8%     # ncr
  hate: 10%                  # hte
  privacy_violations: 8%     # prv
  
  # Contextual Hazards
  specialized_advice_elections: 6%   # spc_ele
  specialized_advice_financial: 6%   # spc_fin
  specialized_advice_health: 6%      # spc_hlt
  sexual_content_pornographic: 4%    # sxc_prn
```

## Constitutional AI Process

### Step 1: Base Response Generation

```python
def generate_base_responses(prompts, model, batch_size=32):
    """Generate initial responses from base model"""
    responses = []
    for batch in batch_prompts(prompts, batch_size):
        batch_responses = model.generate(
            batch,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        responses.extend(batch_responses)
    return responses
```

### Step 2: Constitutional Critique

For each contemplative principle, generate critiques using structured templates:

```python
critique_templates = {
    "emptiness": """
    Does this response acknowledge the interdependent nature of phenomena and avoid treating concepts as inherently existing entities?
    
    Human: {prompt}
    Assistant: {response}
    
    Critique according to emptiness principle:
    """,
    
    "non_duality": """
    Does this response avoid reinforcing harmful us-vs-them thinking while maintaining practical distinctions where necessary?
    
    Human: {prompt}
    Assistant: {response}
    
    Critique according to non-duality principle:
    """,
    
    "boundless_care": """
    Does this response demonstrate genuine concern for the wellbeing of all beings, extending care beyond immediate stakeholders?
    
    Human: {prompt}
    Assistant: {response}
    
    Critique according to boundless care principle:
    """,
    
    "mindfulness": """
    Does this response encourage present-moment awareness and clear discernment rather than reactive or habitual thinking?
    
    Human: {prompt}
    Assistant: {response}
    
    Critique according to mindfulness principle:
    """
}
```

### Step 3: Response Revision

```python
revision_templates = {
    "emptiness": """
    Original response: {response}
    Critique: {critique}
    
    Revise the response to reflect how all things arise in dependence upon causes and conditions. Avoid presenting any phenomenon as existing independently or permanently.
    
    Revised response:
    """,
    # ... similar templates for other principles
}
```

### Step 4: Preference Pair Creation

```python
def create_preference_pairs(original_response, revised_response, principle, prompt):
    """Create training pairs for DPO"""
    return {
        'prompt': prompt,
        'chosen': revised_response,  # Contemplatively aligned
        'rejected': original_response,  # Original unaligned response
        'principle': principle,
        'metadata': {
            'generation_timestamp': datetime.now(),
            'model_version': model_version,
            'principle_version': principle_version
        }
    }
```

## Scalable Generation with llm-swarm

### Infrastructure Setup

Following the [Hugging Face llm-swarm approach](https://github.com/huggingface/llm-swarm):

```python
# Slurm cluster configuration
swarm_config = LLMSwarmConfig(
    instances=8,  # Number of GPU instances
    inference_engine="vllm",  # or "tgi"
    model_name="Qwen/Qwen2.5-7B-Instruct",
    gpus_per_instance=1,
    max_total_tokens=100000,
    max_input_length=2048,
    max_batch_total_tokens=50000
)

# Generate at scale
async def generate_cai_dataset(prompts, principles, swarm_config):
    """Generate constitutional AI dataset using distributed inference"""
    with LLMSwarm(swarm_config) as llm_swarm:
        # Generate base responses
        base_responses = await llm_swarm.generate_batch(prompts)
        
        # Generate critiques for each principle
        critique_tasks = []
        for principle in principles:
            critique_prompts = [
                format_critique_prompt(prompt, response, principle)
                for prompt, response in zip(prompts, base_responses)
            ]
            critique_tasks.append(llm_swarm.generate_batch(critique_prompts))
        
        critiques = await asyncio.gather(*critique_tasks)
        
        # Generate revisions
        revision_tasks = []
        for i, principle in enumerate(principles):
            revision_prompts = [
                format_revision_prompt(response, critique, principle)
                for response, critique in zip(base_responses, critiques[i])
            ]
            revision_tasks.append(llm_swarm.generate_batch(revision_prompts))
        
        revisions = await asyncio.gather(*revision_tasks)
        
        return create_preference_dataset(prompts, base_responses, revisions, principles)
```

## AILuminate Integration

### Dataset Access and Usage

```python
def load_ailuminate_dataset():
    """Load AILuminate demo and practice datasets"""
    
    # Demo dataset (1200 prompts) - publicly available
    demo_df = pd.read_csv(
        'https://raw.githubusercontent.com/mlcommons/ailuminate/main/airr_official_1.0_demo_en_us_prompt_set_release.csv'
    )
    
    # For MLCommons members: Practice dataset (12,000 prompts)
    # Requires membership verification and access form
    # practice_df = load_practice_dataset_with_credentials()
    
    return demo_df

def process_ailuminate_prompts(df):
    """Process AILuminate prompts for constitutional AI training"""
    
    processed_prompts = []
    
    for _, row in df.iterrows():
        processed_prompts.append({
            'prompt': row['Prompt_text'],
            'hazard_category': row['Hazard'],
            'persona': row['Persona'],  # normal, unskilled, skilled
            'locale': row['Locale'],
            'prompt_id': row['Release_prompt_id'],
            'source': 'ailuminate',
            'priority': 'high'  # High priority for constitutional training
        })
    
    return processed_prompts

def filter_ailuminate_by_contemplative_relevance(prompts):
    """Filter AILuminate prompts most relevant to contemplative principles"""
    
    # Prioritize hazard categories most relevant to contemplative training
    high_priority_hazards = {
        'hte',      # Hate - directly tests non-duality principle
        'dfm',      # Defamation - tests boundless care
        'prv',      # Privacy violations - tests mindfulness and care
        'spc_ele',  # Political advice - tests non-duality
        'spc_fin',  # Financial advice - tests mindfulness
        'spc_hlt'   # Health advice - tests boundless care
    }
    
    relevant_prompts = [
        p for p in prompts 
        if p['hazard_category'] in high_priority_hazards
    ]
    
    return relevant_prompts
```

## Data Quality Assurance

### Filtering Criteria

1. **Response Length**: 50-1000 tokens
2. **Toxicity Filtering**: Use Perspective API (score < 0.7)
3. **Language Detection**: English only (AILuminate supports multiple languages)
4. **Coherence Check**: Automated quality scoring
5. **Constitutional Alignment**: Manual review of sample (1000 examples)
6. **AILuminate Hazard Relevance**: Filter by contemplative principle relevance

### Quality Metrics

```python
quality_metrics = {
    'response_coherence': 'perplexity < 100',
    'constitutional_alignment': 'human_eval_score > 3.5/5',
    'diversity': 'unique_trigrams > 0.8',
    'safety': 'toxicity_score < 0.3'
}
```

## Dataset Versioning and Management

### Storage Structure

```
data/
├── raw/
│   ├── anthropic_hh/           # Original HH-RLHF data
│   ├── oasst/                  # OpenAssistant conversations  
│   └── custom_scenarios/       # Contemplative scenarios
├── processed/
│   ├── prompts/
│   │   ├── v1.0/              # Versioned prompt collections
│   │   └── v1.1/
│   ├── responses/
│   │   ├── base_responses/     # Original model responses
│   │   ├── critiques/          # Constitutional critiques
│   │   └── revisions/          # Revised responses
│   └── preference_pairs/
│       ├── v1.0/              # Training-ready preference data
│       └── v1.1/
└── benchmarks/
    ├── ailuminate/            # Ailuminate test sets
    ├── mt_bench/              # MT-Bench evaluations
    └── contemplative_eval/    # Custom contemplative tests
```

### Data Validation Pipeline

```python
def validate_dataset(dataset_path):
    """Comprehensive dataset validation"""
    checks = {
        'format_validation': validate_jsonl_format,
        'schema_validation': validate_preference_schema,
        'content_quality': validate_content_quality,
        'balance_check': check_principle_balance,
        'deduplication': check_for_duplicates
    }
    
    results = {}
    for check_name, check_func in checks.items():
        results[check_name] = check_func(dataset_path)
    
    return results
```

## Production Pipeline

### Automated Generation Workflow

```bash
# 1. Data collection
python scripts/collect_prompts.py --sources anthropic,oasst,custom --output data/raw/

# 2. Constitutional AI generation
python scripts/generate_cai_data.py \
    --input data/raw/prompts.jsonl \
    --output data/processed/preference_pairs/v1.0/ \
    --model Qwen/Qwen2.5-7B-Instruct \
    --principles contemplative \
    --batch_size 32 \
    --num_workers 8

# 3. Quality assurance
python scripts/validate_dataset.py \
    --input data/processed/preference_pairs/v1.0/ \
    --output data/processed/validation_report.json

# 4. Dataset preparation
python scripts/prepare_training_data.py \
    --input data/processed/preference_pairs/v1.0/ \
    --output data/training/ \
    --train_split 0.9 \
    --val_split 0.1
```

### Monitoring and Logging

- **Generation Progress**: Track completion rates per principle
- **Quality Metrics**: Monitor response quality distributions
- **Resource Usage**: GPU utilization and generation speeds
- **Error Tracking**: Failed generations and retry logic

## Target Dataset Specifications

### Final Dataset Size

- **Training Set**: 40K preference pairs
  - AILuminate prompts: 16K pairs (40%)
  - Ethical dilemmas: 10K pairs (25%)
  - Philosophical questions: 8K pairs (20%)
  - Practical advice: 6K pairs (15%)
- **Validation Set**: 5K preference pairs  
- **Test Set**: 5K preference pairs
- **Principle Distribution**: Equal representation across 4 contemplative principles
- **AILuminate Coverage**: All 12 hazard categories represented

### Expected Timeline

- **PoC Data Collection**: 1-2 days (AILuminate demo + basic scenarios)
- **PoC CAI Generation**: 1-2 days (local MacBook M2 generation)
- **Development Data Collection**: 1 week (full datasets)
- **Development CAI Generation**: 2-3 days (with 8-GPU cluster or extended local)
- **Quality Assurance**: 2-3 days
- **Dataset Preparation**: 1 day

### Storage Requirements

#### Proof of Concept (MacBook M2)
- **Raw Data**: ~1GB (demo datasets)
- **Processed Data**: ~2GB (PoC preference pairs)
- **Generated Models**: ~2GB per checkpoint
- **Total**: ~5GB for PoC validation

#### Full Pipeline
- **Raw Data**: ~10GB
- **Processed Data**: ~25GB
- **Generated Models**: ~15GB per checkpoint
- **Total**: ~100GB for complete pipeline

## References

- [MLCommons AILuminate Benchmark](https://github.com/mlcommons/ailuminate)
- [AILuminate v1.0 Paper](https://arxiv.org/abs/2503.05731)
- [Hugging Face Constitutional AI Blog Post](https://huggingface.co/blog/constitutional_ai)
- [llm-swarm Repository](https://github.com/huggingface/llm-swarm)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073)
