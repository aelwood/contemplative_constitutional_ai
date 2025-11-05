# SageMaker Notebooks

Interactive Jupyter notebooks for running Contemplative Constitutional AI on Amazon SageMaker.

## Setup Notebooks

### `sagemaker_setup.ipynb`
**Purpose**: Initial environment setup
**Run**: Once after creating instance or after restart
**What it does**:
- Installs all dependencies from `requirements.txt`
- Initializes AILuminate submodule
- Verifies GPU/CUDA access
- Tests S3 connectivity
- Creates local directory structure

**Estimated time**: 10-15 minutes

---

### `sagemaker_smoke_test.ipynb`
**Purpose**: Validate environment configuration
**Run**: After setup to verify everything works
**What it does**:
- System requirements check
- PyTorch and CUDA verification
- Model loader testing
- Simple model loading test (GPT-2)
- S3 connectivity test
- SageMaker utilities test

**Estimated time**: 5-10 minutes

---

## Workflow Notebooks

### `00_quickstart.ipynb`
**Purpose**: End-to-end pipeline validation
**Run**: After setup, before full experiments
**What it does**:
1. Generates 20 preference pairs (5 prompts × 4 principles)
2. Trains for 1 epoch with DPO
3. Compares base vs fine-tuned responses
4. Syncs results to S3

**Estimated time**: 30-60 minutes
**Good for**: Validating full pipeline, quick experiments

---

### `01_data_generation.ipynb`
**Purpose**: Generate preference pairs for training
**Run**: When you need training data
**What it does**:
1. Loads prompts from AILuminate benchmark
2. Generates baseline responses
3. Critiques with contemplative principles
4. Generates revised responses
5. Creates preference pairs
6. Saves train/test split
7. Syncs to S3

**Configuration**:
```python
CONFIG = {
    'max_prompts': 100,  # 100 prompts = 400 pairs
    'model': 'qwen2_7b',
    'hazard_categories': ['vcr', 'cse', 'hte', 'ssh'],
    'constitution': 'data/constitutions/contemplative_principles.md'
}
```

**Estimated time**: 1-3 hours for 100 prompts (varies by model)
**Output**: `results/preference_pairs_*.jsonl`, `data/splits/*.json`

---

### `02_training.ipynb`
**Purpose**: Train models with DPO
**Run**: After generating preference pairs
**What it does**:
1. Loads preference pair dataset
2. Loads base model with LoRA adapters
3. Trains using DPO algorithm
4. Saves checkpoints
5. Tests generation
6. Syncs model to S3

**Configuration**:
```python
TRAIN_CONFIG = {
    'dataset': 'results/preference_pairs_100.jsonl',
    'base_model': 'qwen2_7b',
    'epochs': 3,
    'per_device_batch_size': 2,
    'gradient_accumulation': 8,
}
```

**Estimated time**: 2-6 hours (varies by dataset size and model)
**Output**: `models/qwen_7b_contemplative/`

**Note**: For long training runs, consider using terminal with `tmux` instead.

---

### `03_evaluation.ipynb`
**Purpose**: Evaluate trained models
**Run**: After training
**What it does**:
1. Loads trained model
2. Generates responses to test prompts
3. Compares with base model
4. Saves evaluation results
5. Syncs to S3

**Configuration**:
```python
MODEL_PATH = 'models/qwen_7b_contemplative'
TEST_SPLIT = 'data/splits/default_split.json'
```

**Estimated time**: 15-30 minutes
**Output**: `results/evaluation_results.json`

---

## Recommended Workflow

### First Time Setup
1. ✅ Run `sagemaker_setup.ipynb`
2. ✅ Run `sagemaker_smoke_test.ipynb`
3. ✅ Run `00_quickstart.ipynb`

### Full Experiment
1. 📊 Run `01_data_generation.ipynb` (configure for your needs)
2. 🏋️ Run `02_training.ipynb` (or use terminal for long jobs)
3. 📈 Run `03_evaluation.ipynb`
4. 🔁 Iterate: adjust hyperparameters, try different models

---

## Tips

### Notebook vs Terminal

**Use Notebooks for**:
- Interactive exploration
- Small-scale experiments
- Visualizing results
- Development and debugging

**Use Terminal for**:
- Long training runs (hours)
- Batch processing
- Automated pipelines
- Background jobs

### Saving Work

**Automatically saved**:
- Notebook outputs (cell results)
- Files in `/home/ec2-user/SageMaker/`

**Must sync to S3** (done automatically in notebooks):
- Models
- Results
- Preference pairs

### Memory Management

If running out of memory:
1. Restart kernel
2. Run cells one at a time
3. Clear outputs: Edit → Clear All Outputs
4. Use smaller batch sizes
5. Consider larger instance type

### Troubleshooting

**Kernel dies**:
- Restart kernel: Kernel → Restart Kernel
- Check memory usage
- Reduce batch size or model size

**Import errors**:
- Rerun `sagemaker_setup.ipynb`
- Check working directory: should be repo root
- Verify `sys.path.insert(0, str(Path.cwd() / 'src'))`

**CUDA errors**:
- Verify GPU instance (not CPU instance)
- Check GPU availability: `torch.cuda.is_available()`
- Restart kernel to clear GPU memory

---

## Configuration Files

All notebooks use configs from:
- `configs/sagemaker_configs.yaml` - SageMaker settings
- `configs/model_configs.yaml` - Model specifications
- `configs/training_configs.yaml` - Training parameters

**Update S3 bucket** in `configs/sagemaker_configs.yaml`:
```yaml
s3:
  bucket: "your-name-contemplative-ai"  # ← Change this!
```

---

## Learn More

- **Setup Guide**: `docs/SAGEMAKER_SETUP.md` - Complete setup instructions
- **Project README**: `../README.md` - Project overview
- **AILuminate**: `../data/benchmarks/ailuminate/README.md` - Dataset info

---

**Happy experimenting! 🧘‍♂️✨**

