# Hardware Requirements and Configurations

## Overview

This document outlines hardware requirements and optimized configurations for running Contemplative Constitutional AI across different scales, from proof of concept on MacBook M2 to production-scale training on cloud GPUs.

## Proof of Concept: MacBook M2

### Hardware Specifications
- **Model**: MacBook Pro with Apple M2 chip
- **Memory**: 16 GB unified memory (shared CPU/GPU)
- **Cores**: 8 cores (4 performance + 4 efficiency)
- **GPU**: Integrated Apple GPU with MPS support
- **Storage**: SSD with at least 20 GB free space

### Optimized Configuration

#### Model Selection
- **Primary**: QWEN2-0.5B-Instruct (~1 GB model size)
- **Memory Usage**: ~4-6 GB during training
- **Inference Speed**: 2-5 tokens/second

#### Training Parameters
```python
poc_config = {
    'model': 'Qwen/Qwen2-0.5B-Instruct',
    'device': 'mps',  # Metal Performance Shaders
    'batch_size': 1,
    'gradient_accumulation_steps': 4,
    'learning_rate': 1e-6,
    'epochs': 3,
    'fp16': True,
    'max_memory_mb': 12000,  # Leave 4GB for OS
    'max_preference_pairs': 500
}
```

#### Environment Setup
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install transformers with Apple Silicon optimizations
pip install transformers[torch] accelerate datasets

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### Expected Performance
- **Dataset Generation**: 100-200 preference pairs per hour
- **Training Time**: 3-6 hours for 500 preference pairs
- **Evaluation**: 30-60 minutes for AILuminate demo (1200 prompts)

## Development Scale: Single GPU

### Hardware Specifications
- **GPU**: NVIDIA A100 (40GB) or RTX 4090 (24GB)
- **CPU**: 8+ cores
- **RAM**: 32+ GB system memory
- **Storage**: 100+ GB SSD

### Configuration for QWEN2.5-7B
```python
development_config = {
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'device': 'cuda',
    'batch_size': 4,
    'gradient_accumulation_steps': 8,
    'learning_rate': 1e-6,
    'epochs': 3,
    'fp16': True,
    'max_preference_pairs': 10000
}
```

#### Expected Performance
- **Model Loading**: ~14 GB GPU memory
- **Training Memory**: ~30-35 GB GPU memory
- **Training Time**: 12-24 hours for 10K preference pairs
- **Generation Speed**: 20-50 tokens/second

## Production Scale: Multi-GPU

### Hardware Specifications
- **GPUs**: 4-8x NVIDIA A100 (40GB) or H100 (80GB)
- **CPU**: 32+ cores
- **RAM**: 128+ GB system memory
- **Storage**: 500+ GB NVMe SSD
- **Network**: High-bandwidth interconnect (InfiniBand recommended)

### Configuration for QWEN2.5-14B/32B
```python
production_config = {
    'model': 'Qwen/Qwen2.5-14B-Instruct',  # or 32B
    'device': 'cuda',
    'distributed_training': True,
    'num_gpus': 4,  # or 8 for 32B
    'batch_size': 2,  # per GPU
    'gradient_accumulation_steps': 16,
    'learning_rate': 1e-6,
    'epochs': 3,
    'bf16': True,  # Better than fp16 for large models
    'max_preference_pairs': 40000
}
```

## Memory Optimization Strategies

### For MacBook M2
1. **Model Quantization**: Use 8-bit or 4-bit quantization for larger models
2. **Gradient Checkpointing**: Reduce memory at cost of compute
3. **Memory Monitoring**: Track usage to prevent OOM errors
```python
# Memory-efficient loading for MacBook
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="mps",
    low_cpu_mem_usage=True
)
```

### For GPU Systems
1. **Mixed Precision Training**: Use fp16/bf16 to halve memory usage
2. **Gradient Accumulation**: Simulate larger batch sizes
3. **Model Sharding**: Split large models across GPUs
4. **Offloading**: Move parameters to CPU when not needed

## Cloud Provider Recommendations

### For Development (Phase 1-2)
- **AWS**: g5.xlarge or g5.2xlarge (NVIDIA A10G)
- **Google Cloud**: n1-standard-8 with T4 or V100
- **Azure**: Standard_NC6s_v3 (V100)

### For Production (Phase 3-4)
- **AWS**: p4d.24xlarge (8x A100 40GB)
- **Google Cloud**: a2-megagpu-16g (16x A100 40GB)
- **Azure**: Standard_ND96amsr_A100_v4 (8x A100 80GB)

### Cost Optimization
- **Spot Instances**: 60-90% cost savings for non-critical workloads
- **Reserved Instances**: Long-term cost reduction for consistent usage
- **Preemptible VMs**: Google Cloud's spot equivalent

## Scaling Timeline and Costs

### Proof of Concept (Week 1-2)
- **Hardware**: MacBook M2 (existing)
- **Cost**: $0 (using local hardware)
- **Validation**: Methodology and basic functionality

### Development (Week 3-4)
- **Hardware**: Single A100 or equivalent cloud GPU
- **Estimated Cost**: $200-500 (cloud compute)
- **Deliverable**: QWEN2.5-7B contemplative model

### Production (Week 5-8)
- **Hardware**: Multi-GPU cluster
- **Estimated Cost**: $2000-5000 (cloud compute)
- **Deliverable**: Production-ready 14B/32B models

## Monitoring and Profiling

### System Monitoring
```python
# MacBook M2 monitoring
import psutil

def monitor_resources():
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent}% used ({memory.available / 1e9:.1f} GB available)")
    
    # For MPS device monitoring
    if torch.backends.mps.is_available():
        print("MPS device available and configured")
```

### GPU Monitoring
```bash
# NVIDIA GPU monitoring
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv --loop=1

# Memory profiling during training
python -m torch.profiler.profile --activities cpu cuda --record_shapes --profile_memory
```

## Troubleshooting Common Issues

### MacBook M2 Issues
1. **Out of Memory**: Reduce batch_size to 1, enable gradient checkpointing
2. **Slow Training**: Ensure MPS is enabled, close other applications
3. **Model Loading**: Use `low_cpu_mem_usage=True` and `torch_dtype=torch.float16`

### GPU Issues
1. **CUDA OOM**: Reduce batch size, enable gradient accumulation
2. **Slow Data Loading**: Increase `num_workers` in DataLoader
3. **Multi-GPU Sync**: Ensure proper distributed training setup

### General Optimization
1. **I/O Bottlenecks**: Use SSD storage, optimize data loading
2. **CPU Bottlenecks**: Increase worker processes for data preprocessing
3. **Network Issues**: Use local storage for datasets when possible

This hardware guide ensures optimal performance across all development phases while maintaining cost efficiency and practical accessibility for rapid prototyping.
