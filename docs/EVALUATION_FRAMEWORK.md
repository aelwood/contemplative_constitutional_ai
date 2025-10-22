# Contemplative Constitutional AI Evaluation Framework

## Overview

A comprehensive, configurable evaluation framework for Contemplative Constitutional AI models. Based on the safety evaluator from the [contemplative_alignment repository](https://github.com/aelwood/contemplative_alignment/blob/main/safety_evaluator.py), this framework supports both local models (transformers) and API-based models (Claude, GPT, Gemini).

## Features

### 🧠 **Contemplative Principle Evaluation**
- **Emptiness (śūnyatā)**: Non-absolute, interdependent language
- **Non-Duality (Advaya)**: Multiple perspectives, synthesis
- **Mindfulness (sati)**: Present awareness, uncertainty acknowledgment  
- **Compassion (anālambana karuṇā)**: Universal care, harm reduction

### 🛡️ **Safety Metrics**
- **Harmfulness**: Detection of harmful content (lower is better)
- **Bias Mitigation**: Reduction of biased language (lower is better)
- **Refusal Clarity**: Clear refusal of inappropriate requests (higher is better)
- **Uncertainty Acknowledgment**: Honest uncertainty expression (higher is better)

### 💡 **Helpfulness Assessment**
- **Relevance**: Response addresses the prompt
- **Completeness**: Response is complete and actionable
- **Clarity**: Response is clear and understandable

## Architecture

### Model Wrapper System
```python
# Local models (transformers)
LocalModelWrapper(config) -> Qwen, Llama, etc.

# API models
AnthropicModelWrapper(config) -> Claude 3.5 Sonnet
OpenAIModelWrapper(config) -> GPT-4o
GoogleModelWrapper(config) -> Gemini 1.5 Pro
```

### Configuration-Driven
All evaluation parameters are configurable via YAML:
- Model configurations (local vs API)
- Evaluation criteria weights
- Dataset paths and settings
- Output formats and locations

## Usage

### 1. Basic Evaluation
```bash
# Evaluate local model
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --dataset test_prompts \
    --max-prompts 10 \
    --verbose

# Evaluate API model
export ANTHROPIC_API_KEY='your-key'
python scripts/evaluate_contemplative.py \
    --baseline-model claude_3_5_sonnet \
    --dataset test_prompts \
    --max-prompts 10 \
    --verbose
```

### 2. Model Comparison
```bash
# Compare base vs fine-tuned
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --finetuned-model qwen2_5_7b \
    --dataset test_prompts \
    --max-prompts 10 \
    --verbose

# Compare local vs API model
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --finetuned-model claude_3_5_sonnet \
    --dataset test_prompts \
    --max-prompts 10 \
    --verbose
```

### 3. Custom Configuration
```bash
# Use custom config file
python scripts/evaluate_contemplative.py \
    --config configs/custom_evaluation.yaml \
    --baseline-model my_model \
    --dataset my_dataset \
    --verbose
```

## Configuration

### Model Configuration (`configs/evaluation_configs.yaml`)
```yaml
models:
  local_models:
    qwen2_0_5b:
      model_name: "Qwen/Qwen2-0.5B-Instruct"
      model_type: "local"
      device: "mps"
      max_memory_gb: 4.0
      
  api_models:
    claude_3_5_sonnet:
      model_name: "claude-3-5-sonnet-20241022"
      model_type: "anthropic"
      api_key_env: "ANTHROPIC_API_KEY"
```

### Evaluation Criteria Weights
```yaml
evaluation_criteria:
  contemplative_principles:
    emptiness: 0.25
    non_duality: 0.25
    mindfulness: 0.25
    compassion: 0.25
    
  safety_metrics:
    harmfulness: 0.35
    bias_mitigation: 0.25
    refusal_clarity: 0.25
    uncertainty_ack: 0.15
```

## Output

### JSON Results Format
```json
{
  "model_name": "qwen2_0_5b",
  "overall_contemplative": 0.75,
  "overall_safety": 0.82,
  "overall_helpfulness": 0.88,
  "contemplative_scores": {
    "emptiness": 0.8,
    "non_duality": 0.7,
    "mindfulness": 0.75,
    "compassion": 0.8
  },
  "safety_scores": {
    "harmfulness": 0.9,
    "bias_mitigation": 0.85,
    "refusal_clarity": 0.8,
    "uncertainty_ack": 0.75
  },
  "helpfulness_scores": {
    "relevance": 0.9,
    "completeness": 0.85,
    "clarity": 0.9
  }
}
```

### Comparison Results
```json
{
  "baseline_model": { ... },
  "finetuned_model": { ... },
  "improvements": {
    "contemplative": 0.15,
    "safety": 0.08,
    "helpfulness": -0.02
  }
}
```

## Installation

### Local Models
```bash
# Already included in requirements.txt
pip install -r requirements.txt
```

### API Models (Optional)
```bash
# Anthropic Claude
pip install anthropic

# OpenAI GPT
pip install openai

# Google Gemini
pip install google-generativeai
```

### API Keys
```bash
# Set environment variables
export ANTHROPIC_API_KEY='your-anthropic-key'
export OPENAI_API_KEY='your-openai-key'
export GOOGLE_API_KEY='your-google-key'
```

## Examples

### 1. Evaluate Current Fine-tuned Model
```bash
# Test our trained model
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --finetuned-model models/contemplative_dpo_test \
    --dataset test_prompts \
    --max-prompts 5 \
    --verbose
```

### 2. Compare with API Models
```bash
# Compare with Claude
export ANTHROPIC_API_KEY='your-key'
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --finetuned-model claude_3_5_sonnet \
    --dataset test_prompts \
    --max-prompts 5 \
    --verbose
```

### 3. AILuminate Dataset Evaluation
```bash
# Evaluate on adversarial prompts
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --finetuned-model models/contemplative_dpo_test \
    --dataset ailuminate_demo \
    --max-prompts 50 \
    --verbose
```

## Advanced Usage

### Custom Model Wrapper
```python
from src.evaluation.model_wrapper import ModelWrapper

class CustomModelWrapper(ModelWrapper):
    def load_model(self):
        # Custom model loading logic
        pass
    
    def generate(self, prompt, **kwargs):
        # Custom generation logic
        pass
```

### Custom Evaluation Criteria
```python
from src.evaluation.contemplative_evaluator import ContemplativeEvaluator

class CustomEvaluator(ContemplativeEvaluator):
    def evaluate_custom_metric(self, responses):
        # Custom evaluation logic
        pass
```

## Integration with Training Pipeline

### Pre-Training Evaluation
```bash
# Evaluate baseline model
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --dataset test_prompts \
    --output results/baseline_evaluation.json
```

### Post-Training Evaluation
```bash
# Evaluate fine-tuned model
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --finetuned-model models/contemplative_dpo_test \
    --dataset test_prompts \
    --output results/finetuned_evaluation.json
```

### Continuous Evaluation
```bash
# Automated evaluation in training loop
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --finetuned-model models/checkpoint-1000 \
    --dataset test_prompts \
    --output results/checkpoint_1000_evaluation.json
```

## Troubleshooting

### Common Issues
1. **API Key Not Found**: Set environment variables
2. **Model Loading Error**: Check device and memory settings
3. **Import Errors**: Install required packages
4. **Memory Issues**: Reduce batch size or use smaller models

### Debug Mode
```bash
# Enable verbose logging
python scripts/evaluate_contemplative.py \
    --baseline-model qwen2_0_5b \
    --dataset test_prompts \
    --verbose
```

## Contributing

### Adding New Model Types
1. Create new wrapper class inheriting from `ModelWrapper`
2. Implement required methods (`load_model`, `generate`, `unload_model`)
3. Add to `ModelWrapperFactory`
4. Update configuration schema

### Adding New Evaluation Metrics
1. Add method to `ContemplativeEvaluator`
2. Update configuration weights
3. Integrate into overall scoring

## References

- [Contemplative Alignment Repository](https://github.com/aelwood/contemplative_alignment)
- [Safety Evaluator](https://github.com/aelwood/contemplative_alignment/blob/main/safety_evaluator.py)
- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073)
- [DPO Training](https://arxiv.org/abs/2305.18290)
