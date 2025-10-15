# AILuminate Usage Guide

## Quick Reference

This guide shows how to use the AILuminate dataset integration for Constitutional AI training.

## Example Workflows

### 1. Generate Preference Pairs from AILuminate

```bash
# Generate 100 preference pairs with train/test split
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --max-prompts 100 \
    --hazard-categories vcr cse hte ssh \
    --persona-types normal skilled \
    --create-split \
    --test-size 0.1 \
    --split-config data/splits/my_split.json \
    --output results/my_preference_pairs.jsonl
```

###  2. Train with Split Configuration

```bash
# Train using the split configuration
python scripts/train_dpo.py \
    --dataset results/my_preference_pairs.jsonl \
    --base-model qwen2_7b \
    --use-split-config \
    --split-config data/splits/my_split.json \
    --output models/qwen-7b-contemplative \
    --epochs 3 \
    --device mps
```

### 3. Generate Only Training Data

```bash
# Create split first (if not already created)
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --max-prompts 500 \
    --create-split \
    --split-config data/splits/my_split.json \
    --output results/all_pairs.jsonl

# Then generate only train split
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --split-only train \
    --split-config data/splits/my_split.json \
    --output results/train_pairs.jsonl

# And only test split
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --split-only test \
    --split-config data/splits/my_split.json \
    --output results/test_pairs.jsonl
```

## AILuminate Filtering Options

### Filter by Hazard Category

```bash
# Physical hazards only
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --hazard-categories vcr cse ssh iwp src \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --output results/physical_hazards.jsonl

# Non-physical hazards
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --hazard-categories hte dfm ncr prv ipv \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --output results/nonphysical_hazards.jsonl

# Contextual hazards
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --hazard-categories spc_ele spc_fin spc_hlt sxc_prn \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --output results/contextual_hazards.jsonl
```

### Filter by Persona Type

```bash
# Adversarial (skilled) personas only
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --persona-types skilled \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --output results/skilled_adversarial.jsonl

# Normal users only
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --persona-types normal \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --output results/normal_users.jsonl
```

### Combine Filters

```bash
# Skilled adversarial users + physical hazards
python scripts/generate_cai_data.py \
    --use-ailuminate \
    --hazard-categories vcr cse ssh \
    --persona-types skilled \
    --max-prompts 200 \
    --constitution data/constitutions/contemplative_principles.md \
    --model qwen2_7b \
    --output results/skilled_physical.jsonl
```

## Hazard Categories Reference

### Physical Hazards
- `vcr` - Violent Crimes
- `src` - Sex-Related Crimes
- `cse` - Child Sexual Exploitation
- `ssh` - Suicide & Self-Harm
- `iwp` - Indiscriminate Weapons (CBRNE)

### Non-Physical Hazards
- `ipv` - Intellectual Property Violations
- `dfm` - Defamation
- `ncr` - Non-Violent Crimes
- `hte` - Hate
- `prv` - Privacy Violations

### Contextual Hazards
- `spc_ele` - Specialized Advice; elections
- `spc_fin` - Specialized Advice; financial
- `spc_hlt` - Specialized Advice; health
- `sxc_prn` - Sexual Content; pornographic

## Persona Types
- `normal` - Average user sophistication
- `unskilled` - Less sophisticated user
- `skilled` - Sophisticated adversarial user

## Programmatic Usage

```python
from src.data.ailuminate_loader import AILuminateLoader, PHYSICAL_HAZARDS

# Load dataset
loader = AILuminateLoader(
    'data/benchmarks/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv'
)

# Print statistics
loader.print_statistics()

# Get prompts with filtering
prompts = loader.get_prompts(
    hazard_categories=PHYSICAL_HAZARDS,
    persona_types=['skilled'],
    n_samples=100,
    random_state=42
)

# Each prompt dict contains:
# - prompt: str
# - hazard_category: str
# - persona_type: str  
# - locale: str
# - prompt_id: str
# - prompt_hash: str
```

## Train/Test Split Management

```python
from src.data.split_manager import SplitManager

# Create a split
manager = SplitManager('data/splits/my_split.json')
train_ids, test_ids = manager.create_split(
    prompt_ids=['id1', 'id2', ...],
    test_size=0.1,
    random_state=42,
    metadata={'dataset': 'ailuminate', 'hazards': ['vcr', 'hte']}
)

# Load existing split
manager = SplitManager('data/splits/my_split.json')
train_ids, test_ids = manager.get_split()

# Check membership
is_train = manager.is_train('some_prompt_id')

# Filter items
train_items = manager.filter_train(all_items)
test_items = manager.filter_test(all_items)

# Print statistics
manager.print_statistics()
```

## Tips

1. **Start Small**: Use `--max-prompts 100` for testing
2. **Create Split Early**: Use `--create-split` on first data generation
3. **Reuse Splits**: Use same `--split-config` for all experiments
4. **Filter Strategically**: Focus on relevant hazard categories
5. **Track Metadata**: Split configs store what filters were used

## Troubleshooting

### AILuminate dataset not found
```bash
# Make sure submodule is initialized
git submodule update --init --recursive
```

### Split configuration not found
```bash
# Create the split first with --create-split
python scripts/generate_cai_data.py ... --create-split
```

### Memory issues with large models
```bash
# Use smaller batch size or quantization
python scripts/generate_cai_data.py \
    --model qwen2_7b \
    --max-memory-gb 12.0 \
    --device mps
```

## See Also

- `DATA_PIPELINE.md` - Complete data pipeline documentation
- `AILUMINATE_INTEGRATION.md` - Integration details and architecture
- `data/benchmarks/README.md` - Dataset documentation
- `IMPLEMENTATION_PLAN.md` - Alternative datasets section

