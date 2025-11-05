# SageMaker Setup Guide

Complete guide for running Contemplative Constitutional AI on Amazon SageMaker.

## Overview

This guide walks you through setting up and using SageMaker notebooks for Constitutional AI experiments. SageMaker provides:
- GPU access (NVIDIA T4, A10G, V100)
- Persistent storage via S3
- JupyterLab interface
- Pay-per-use pricing

## Prerequisites

- AWS Account
- Basic familiarity with Jupyter notebooks
- S3 bucket for storing results

## Part 1: AWS Setup

### 1.1 Create S3 Bucket

1. Go to AWS Console → S3
2. Click "Create bucket"
3. Bucket name: `your-name-contemplative-ai` (must be unique)
4. Region: `us-east-1` (or your preferred region)
5. Keep default settings, click "Create bucket"

### 1.2 Create IAM Role

1. Go to AWS Console → IAM → Roles
2. Click "Create role"
3. Select "AWS Service" → "SageMaker"
4. Add permissions:
   - `AmazonSageMakerFullAccess` (already attached)
   - `AmazonS3FullAccess` (add this)
5. Name: `SageMaker-ContemplativeAI-Role`
6. Click "Create role"

### 1.3 Create SageMaker Notebook Instance

1. Go to AWS Console → SageMaker → Notebook instances
2. Click "Create notebook instance"

**Configuration:**
- **Name**: `contemplative-ai-dev`
- **Instance type**: `ml.g5.2xlarge` (recommended)
  - Alternative: `ml.g4dn.xlarge` (cheaper, smaller GPU)
  - Alternative: `ml.p3.2xlarge` (V100 GPU)
- **Platform identifier**: Amazon Linux 2, Jupyter Lab 3
- **IAM role**: Select `SageMaker-ContemplativeAI-Role`
- **Root access**: Enabled
- **Volume size**: 100 GB minimum

**Git repositories (optional but recommended):**
- **Repository**: `https://github.com/yourusername/contemplative_constitutional_ai.git`
- **Branch**: `main`

3. Click "Create notebook instance"
4. Wait 5-10 minutes for instance to be "InService"

## Part 2: Initial Setup

### 2.1 Access JupyterLab

1. In SageMaker console, find your instance
2. Click "Open JupyterLab"
3. JupyterLab opens in a new tab

### 2.2 Clone Repository (if not done via Git integration)

Open a terminal in JupyterLab (File → New → Terminal):

```bash
cd /home/ec2-user/SageMaker/
git clone https://github.com/yourusername/contemplative_constitutional_ai.git
cd contemplative_constitutional_ai
```

### 2.3 Initialize Submodules

```bash
git submodule update --init --recursive
```

This downloads the AILuminate benchmark dataset.

### 2.4 Configure S3 Bucket

Edit `configs/sagemaker_configs.yaml`:

```bash
nano configs/sagemaker_configs.yaml
```

Update the bucket name:
```yaml
s3:
  bucket: "your-name-contemplative-ai"  # Change this!
```

Save (Ctrl+X, Y, Enter)

### 2.5 Run Setup Notebook

1. Navigate to `notebooks/` in JupyterLab file browser
2. Open `sagemaker_setup.ipynb`
3. Run all cells (Run → Run All Cells)
4. Verify all cells show ✅

This installs all dependencies and configures the environment.

### 2.6 Run Smoke Test

1. Open `notebooks/sagemaker_smoke_test.ipynb`
2. Run all cells
3. Verify all tests pass

If any tests fail, check the error messages and troubleshoot before proceeding.

## Part 3: Workflow

### 3.1 Quick Validation (Recommended First Step)

Run the quickstart to verify everything works:

1. Open `notebooks/00_quickstart.ipynb`
2. Run all cells
3. **Runtime**: ~30-60 minutes
4. **Result**: Small trained model demonstrating full pipeline

### 3.2 Full Workflow

For production experiments, use the individual workflow notebooks:

#### Step 1: Generate Preference Pairs

1. Open `notebooks/01_data_generation.ipynb`
2. Configure parameters:
   - `max_prompts`: Number of prompts (100 = 400 pairs with 4 principles)
   - `model`: Model for generation (qwen2_0_5b, qwen2_7b, etc.)
   - `hazard_categories`: AILuminate categories to include
3. Run all cells
4. **Runtime**: 1-3 hours for 100 prompts (depends on model size)

#### Step 2: Train Model

1. Open `notebooks/02_training.ipynb`
2. Configure parameters:
   - `dataset`: Path to generated pairs
   - `base_model`: Model to fine-tune
   - `epochs`: Number of training epochs (3 recommended)
3. Run all cells
4. **Runtime**: 2-6 hours (depends on dataset size and model)

**Alternative: Terminal Training**

For long training runs, use terminal to avoid notebook timeout:

```bash
cd /home/ec2-user/SageMaker/contemplative_constitutional_ai

python scripts/train_dpo.py \
    --dataset results/preference_pairs_100.jsonl \
    --base-model qwen2_7b \
    --output models/qwen_7b_contemplative \
    --epochs 3 \
    --use-split-config \
    --device cuda
```

Use `tmux` or `screen` to keep training running if you disconnect:

```bash
# Start tmux session
tmux new -s training

# Run training command
python scripts/train_dpo.py ...

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

#### Step 3: Evaluate Model

1. Open `notebooks/03_evaluation.ipynb`
2. Configure model path
3. Run evaluation
4. Results saved to S3 automatically

## Part 4: Working with S3

### 4.1 Automatic Syncing

The notebooks automatically sync results to S3 when you run them:
- Preference pairs → `s3://bucket/results/preference_pairs/`
- Models → `s3://bucket/models/`
- Evaluations → `s3://bucket/results/evaluations/`

### 4.2 Manual S3 Operations

**Upload to S3:**
```python
from utils.sagemaker_utils import sync_to_s3
sync_to_s3('local/path', 's3://bucket/path')
```

**Download from S3:**
```python
from utils.sagemaker_utils import sync_from_s3
sync_from_s3('s3://bucket/path', 'local/path')
```

**Using AWS CLI:**
```bash
# Upload
aws s3 cp local_file.txt s3://your-bucket/path/

# Download
aws s3 cp s3://your-bucket/path/file.txt ./

# Sync directory
aws s3 sync results/ s3://your-bucket/results/
```

## Part 5: Cost Management

### 5.1 Instance Costs (Approximate)

- **ml.g4dn.xlarge**: $0.74/hour (~$18/day)
- **ml.g5.2xlarge**: $1.21/hour (~$29/day)
- **ml.p3.2xlarge**: $3.83/hour (~$92/day)

### 5.2 Stop Instance When Not in Use

**Manually stop:**
1. Go to SageMaker console
2. Select your instance
3. Actions → Stop
4. **Important**: This doesn't delete your files!

**Auto-stop on idle:**
1. In instance settings
2. Set idle timeout: 60 minutes
3. Instance stops automatically if idle

**Restart:**
- Click "Start" in console
- Rerun `sagemaker_setup.ipynb` to reinstall packages

### 5.3 Storage Costs

- **S3**: ~$0.023/GB/month
- **EBS (notebook volume)**: ~$0.10/GB/month

### 5.4 Cost Saving Tips

1. **Use spot instances** for non-critical workloads (60-90% discount)
2. **Stop instance** when not actively using
3. **Use smaller models** for development (qwen2_0_5b vs qwen2_7b)
4. **Delete old checkpoints** to save storage
5. **Use ml.g5** instances (better price/performance than ml.p3)

## Part 6: Troubleshooting

### GPU Not Detected

**Problem**: CUDA not available
**Solution**:
- Verify instance type has GPU (ml.g5.2xlarge, not ml.t3.medium)
- Restart kernel
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

### S3 Access Denied

**Problem**: Cannot read/write S3
**Solution**:
- Check IAM role has S3 permissions
- Verify bucket name is correct in config
- Check bucket region matches instance region

### Out of Memory

**Problem**: CUDA out of memory during training/generation
**Solution**:
- Reduce batch size
- Use smaller model
- Enable gradient checkpointing
- Use larger instance type

### Package Installation Errors

**Problem**: Pip install fails
**Solution**:
- Check internet connectivity
- Try: `pip install --upgrade pip`
- Restart kernel and rerun setup notebook

### Git Submodule Issues

**Problem**: AILuminate data not found
**Solution**:
```bash
cd /home/ec2-user/SageMaker/contemplative_constitutional_ai
git submodule update --init --recursive
```

## Part 7: Advanced Usage

### 7.1 Using Different Instance Types

**For 0.5B-1.5B models:**
- `ml.g4dn.xlarge` (T4, 16GB GPU)
- Cost-effective for small models

**For 7B models:**
- `ml.g5.2xlarge` (A10G, 24GB GPU) ← **Recommended**
- Good balance of cost and performance

**For 14B+ models:**
- `ml.g5.4xlarge` (A10G, 24GB GPU) or
- `ml.p3.8xlarge` (4x V100, 64GB total)

### 7.2 Running Scripts from Terminal

All workflows can run from terminal:

```bash
# Data generation
python scripts/generate_cai_data.py --use-ailuminate --model qwen2_7b --max-prompts 100 ...

# Training
python scripts/train_dpo.py --dataset results/pairs.jsonl --base-model qwen2_7b ...

# Evaluation
python scripts/evaluate_contemplative.py --model models/trained_model ...
```

### 7.3 Custom Configurations

Create experiment-specific configs:

```bash
cp configs/sagemaker_configs.yaml configs/experiment_001.yaml
# Edit experiment_001.yaml
```

Load in notebook:
```python
with open('configs/experiment_001.yaml') as f:
    config = yaml.safe_load(f)
```

### 7.4 Monitoring

**GPU Utilization:**
```bash
watch -n 1 nvidia-smi
```

**Training logs:**
```bash
tail -f logs/training.log
```

## Part 8: Best Practices

1. **Always stop instances** when done for the day
2. **Sync to S3 regularly** - local storage is ephemeral on stop
3. **Use version control** for code changes
4. **Document experiments** in notebook markdown cells
5. **Test with small datasets** before large runs
6. **Monitor costs** in AWS Billing console
7. **Use meaningful names** for models and experiments
8. **Keep checkpoints** for recovery
9. **Run quickstart first** to validate setup
10. **Use terminal for long jobs** with tmux/screen

## Part 9: Getting Help

### Check Logs

Notebook logs: Check cell outputs
Training logs: `models/*/training_args.bin` and output logs
AWS logs: CloudWatch (if enabled)

### Common Resources

- **SageMaker docs**: https://docs.aws.amazon.com/sagemaker/
- **Project README**: `README.md`
- **AILuminate docs**: `data/benchmarks/ailuminate/README.md`

### Support Channels

- GitHub Issues for code problems
- AWS Support for infrastructure issues

## Appendix: Quick Reference

### Instance Recommendations

| Use Case | Instance | GPU | Cost/hr | Best For |
|----------|----------|-----|---------|----------|
| Testing | ml.g4dn.xlarge | T4 | $0.74 | 0.5B-1.5B models |
| Development | ml.g5.2xlarge | A10G | $1.21 | 7B models (recommended) |
| Production | ml.g5.4xlarge | A10G | $2.03 | 7B-14B models |
| Large models | ml.p3.8xlarge | 4x V100 | $14.69 | 14B+ models |

### Essential Commands

```bash
# Navigate to project
cd /home/ec2-user/SageMaker/contemplative_constitutional_ai

# Update code
git pull origin main

# Activate Python environment
# (dependencies installed globally in SageMaker)

# Check GPU
nvidia-smi

# Start tmux session
tmux new -s mysession

# Detach from tmux
Ctrl+B, then D

# Reattach to tmux
tmux attach -t mysession

# S3 sync
aws s3 sync results/ s3://bucket/results/
```

### File Locations

- **Notebooks**: `/home/ec2-user/SageMaker/contemplative_constitutional_ai/notebooks/`
- **Scripts**: `/home/ec2-user/SageMaker/contemplative_constitutional_ai/scripts/`
- **Results**: `/home/ec2-user/SageMaker/contemplative_constitutional_ai/results/`
- **Models**: `/home/ec2-user/SageMaker/contemplative_constitutional_ai/models/`

---

**You're all set! Start with `00_quickstart.ipynb` to validate your setup, then move on to full-scale experiments.**

