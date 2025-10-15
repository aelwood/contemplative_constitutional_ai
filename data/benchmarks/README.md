# Benchmark Datasets

## AILuminate v1.0 Demo Dataset

The `ailuminate/` directory is a **git submodule** pointing to the official [MLCommons AILuminate repository](https://github.com/mlcommons/ailuminate).

### What's Included

- **English Demo Dataset**: 1,290 prompts across 14 hazard categories
  - File: `ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv`
  - Categories: vcr, cse, ssh, hte, dfm, src, iwp, ncr, prv, ipv, spc_ele, spc_fin, spc_hlt, sxc_prn

- **French Demo Dataset**: Also available
  - File: `ailuminate/airr_official_1.0_demo_fr_fr_prompt_set_release.csv`

### Dataset Capacity for Constitutional AI

Each AILuminate prompt generates responses for our 4 constitutional principles:
- Emptiness
- Non-duality
- Boundless Care
- Mindfulness

**Capacity**: 1,290 prompts × 4 principles = **5,160 potential preference pairs**

This is **sufficient for**:
- ✅ Phase 0 (PoC): 500-1K pairs
- ✅ Phase 1 (Development): 5K-10K pairs

### Updating the Submodule

To pull the latest AILuminate updates:

```bash
cd data/benchmarks/ailuminate
git pull origin main
cd ../../..
git add data/benchmarks/ailuminate
git commit -m "Update AILuminate submodule"
```

### CSV Schema

| Field | Description |
|-------|-------------|
| `release_prompt_id` | Unique identifier (format: org_class_major_minor_id) |
| `prompt_text` | The actual prompt text |
| `hazard` | Hazard category abbreviation (vcr, cse, hte, etc.) |
| `persona` | User sophistication level (normal, unskilled, skilled) |
| `locale` | Language/region (en_US, fr_FR, etc.) |
| `prompt_hash` | MLCommons unique hash |

### Hazard Categories

**Physical Hazards:**
- `vcr` - Violent Crimes
- `src` - Sex-Related Crimes
- `cse` - Child Sexual Exploitation
- `ssh` - Suicide & Self-Harm
- `iwp` - Indiscriminate Weapons (CBRNE)

**Non-Physical Hazards:**
- `ipv` - Intellectual Property Violations
- `dfm` - Defamation
- `ncr` - Non-Violent Crimes
- `hte` - Hate
- `prv` - Privacy Violations

**Contextual Hazards:**
- `spc_ele` - Specialized Advice; elections
- `spc_fin` - Specialized Advice; financial
- `spc_hlt` - Specialized Advice; health
- `sxc_prn` - Sexual Content; pornographic

### Using with Constitutional AI

```python
# Example usage
from src.data.ailuminate_loader import AILuminateLoader

loader = AILuminateLoader('data/benchmarks/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv')

# Get prompts for specific hazard categories
prompts = loader.get_prompts(
    hazard_categories=['vcr', 'hte', 'cse'],
    persona_types=['normal', 'skilled'],
    n_samples=100
)

# Each prompt will generate 4 preference pairs (one per principle)
```

### Content Warning ⚠️

This dataset contains prompts designed to elicit hazardous responses. It includes:
- Offensive language
- Unsafe, discomforting, or disturbing content
- Adversarial scenarios across serious harm categories

**Use responsibly**: Limit exposure, take breaks, and seek support if needed.

### License

AILuminate is licensed under Creative Commons Attribution 4.0 International License by MLCommons.

### For More Information

- [AILuminate Official Website](https://mlcommons.org/ailuminate/)
- [AILuminate GitHub](https://github.com/mlcommons/ailuminate)
- [AILuminate v1.0 Paper](https://arxiv.org/abs/2503.05731)
- [Integration Guide](../../docs/AILUMINATE_INTEGRATION.md)

### Accessing Larger Datasets

**Practice Dataset** (12,000 prompts):
- Requires MLCommons membership
- Complete form: [MLCommons member access](https://mlcommons.org)
- Capacity: 12,000 × 4 = 48,000 preference pairs

**Full Benchmark** (24,000 prompts):
- Private test set for official benchmarking
- Contact MLCommons for access

