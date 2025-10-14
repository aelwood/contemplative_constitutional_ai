"""
DPO (Direct Preference Optimization) trainer for Contemplative Constitutional AI.
Optimized for Apple Silicon (MPS) and CUDA devices.
"""

import os
import json
import torch
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

# Core ML libraries
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# TRL for DPO
try:
    from trl import DPOTrainer, DPOConfig
    TRL_AVAILABLE = True
except ImportError:
    logging.warning("TRL not available, using basic trainer implementation")
    TRL_AVAILABLE = False

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training."""
    
    # Model and data
    base_model_key: str = 'qwen2_0_5b'
    dataset_path: str = None
    output_dir: str = './models/contemplative_dpo'
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    beta: float = 0.1  # DPO regularization parameter
    
    # Optimization
    optimizer: str = 'adamw_torch'
    lr_scheduler_type: str = 'cosine'
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Memory and device
    fp16: bool = True
    bf16: bool = False
    device: Optional[str] = None
    max_memory_gb: float = 12.0
    dataloader_num_workers: int = 0
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'eval_loss'
    
    # DPO specific
    max_length: int = 512
    max_prompt_length: int = 256
    remove_unused_columns: bool = False
    
    def __post_init__(self):
        """Set default LoRA target modules if not specified."""
        if self.lora_target_modules is None:
            # Default for Qwen2 models
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class DPOTrainer_Custom:
    """Custom DPO trainer with Apple Silicon optimization."""
    
    def __init__(
        self,
        config: DPOTrainingConfig,
        model_loader: Optional[ModelLoader] = None
    ):
        """
        Initialize the DPO trainer.
        
        Args:
            config: Training configuration
            model_loader: Pre-configured ModelLoader instance
        """
        self.config = config
        
        # Set up model loader
        if model_loader is None:
            model_loader = ModelLoader()
        self.model_loader = model_loader
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.device = None
        
        # Training components
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def load_model_and_tokenizer(self) -> None:
        """Load the base model and tokenizer."""
        logger.info(f"Loading base model: {self.config.base_model_key}")
        
        try:
            self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(
                model_key=self.config.base_model_key,
                device=self.config.device,
                max_memory_gb=self.config.max_memory_gb
            )
            
            self.device = self.model.device if hasattr(self.model, 'device') else 'cpu'
            logger.info(f"Model loaded on device: {self.device}")
            
            # Create reference model (copy of base model for DPO)
            self.ref_model = type(self.model)(self.model.config)
            self.ref_model.load_state_dict(self.model.state_dict())
            if hasattr(self.model, 'device'):
                self.ref_model = self.ref_model.to(self.model.device)
            
            # Apply LoRA if enabled
            if self.config.use_lora:
                self._apply_lora()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_lora(self) -> None:
        """Apply LoRA adaptation to the model."""
        logger.info("Applying LoRA configuration")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self, dataset_path: str, eval_split: float = 0.1) -> None:
        """
        Load preference pairs dataset from JSONL file.
        
        Args:
            dataset_path: Path to JSONL file with preference pairs
            eval_split: Fraction of data to use for evaluation
        """
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Load preference pairs
        preference_pairs = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    preference_pairs.append(data)
        
        logger.info(f"Loaded {len(preference_pairs)} preference pairs")
        
        # Convert to DPO format
        dpo_data = []
        for pair in preference_pairs:
            dpo_example = {
                'prompt': pair['prompt'],
                'chosen': pair['chosen'],
                'rejected': pair['rejected'],
                'principle': pair.get('principle', ''),
            }
            dpo_data.append(dpo_example)
        
        # Split into train/eval
        split_idx = int(len(dpo_data) * (1 - eval_split))
        train_data = dpo_data[:split_idx]
        eval_data = dpo_data[split_idx:]
        
        # Create datasets
        self.train_dataset = Dataset.from_list(train_data)
        self.eval_dataset = Dataset.from_list(eval_data) if eval_data else None
        
        logger.info(f"Created train dataset with {len(train_data)} examples")
        if self.eval_dataset:
            logger.info(f"Created eval dataset with {len(eval_data)} examples")
    
    def _tokenize_function(self, examples):
        """Tokenize examples for DPO training."""
        # Tokenize prompts, chosen, and rejected responses
        prompts = examples['prompt']
        chosen = examples['chosen']
        rejected = examples['rejected']
        
        # Tokenize each component
        tokenized = {
            'prompt': self.tokenizer(prompts, truncation=True, max_length=self.config.max_prompt_length),
            'chosen': self.tokenizer(chosen, truncation=True, max_length=self.config.max_length),
            'rejected': self.tokenizer(rejected, truncation=True, max_length=self.config.max_length)
        }
        
        return tokenized
    
    def prepare_training(self) -> None:
        """Prepare training arguments and trainer."""
        logger.info("Preparing training configuration")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Disable mixed precision on non-CUDA devices to avoid accelerator errors
        device_str = str(self.device) if self.device is not None else "cpu"
        if not device_str.startswith("cuda"):
            if self.config.fp16:
                logger.info("Disabling fp16 for device %s", device_str)
            self.config.fp16 = False
            self.config.bf16 = False

        if TRL_AVAILABLE:
            # Use TRL DPOTrainer
            self._prepare_trl_trainer()
        else:
            # Use custom implementation
            self._prepare_custom_trainer()
    
    def _prepare_trl_trainer(self) -> None:
        """Prepare TRL DPOTrainer."""
        # Training arguments
        training_args = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            remove_unused_columns=self.config.remove_unused_columns,
            beta=self.config.beta,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            dataloader_num_workers=self.config.dataloader_num_workers,
        )
        
        # Create DPO trainer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
    
    def _prepare_custom_trainer(self) -> None:
        """Prepare custom trainer implementation."""
        logger.warning("Using custom trainer - TRL not available")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            dataloader_num_workers=self.config.dataloader_num_workers,
        )
        
        # Tokenize datasets
        if self.train_dataset:
            self.train_dataset = self.train_dataset.map(self._tokenize_function, batched=True)
        if self.eval_dataset:
            self.eval_dataset = self.eval_dataset.map(self._tokenize_function, batched=True)
        
        # Create basic trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
    
    def train(self) -> None:
        """Run the training loop."""
        logger.info("Starting DPO training")
        
        # Save configuration
        config_path = Path(self.config.output_dir) / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False)
        
        # Train the model
        try:
            train_result = self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            
            # Save training metrics
            metrics_path = Path(self.config.output_dir) / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            logger.info(f"Training completed. Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model."""
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset available")
            return {}
        
        logger.info("Running evaluation")
        eval_result = self.trainer.evaluate()
        
        # Save evaluation metrics
        metrics_path = Path(self.config.output_dir) / "eval_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(eval_result, f, indent=2)
        
        return eval_result
    
    def save_model(self, output_path: Optional[str] = None) -> None:
        """Save the trained model."""
        if output_path is None:
            output_path = self.config.output_dir
        
        logger.info(f"Saving model to: {output_path}")
        self.trainer.save_model(output_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)


def train_dpo_model(
    dataset_path: str,
    base_model_key: str = 'qwen2_0_5b',
    output_dir: str = './models/contemplative_dpo',
    config_overrides: Optional[Dict[str, Any]] = None
) -> DPOTrainer_Custom:
    """
    Convenience function to train a DPO model.
    
    Args:
        dataset_path: Path to JSONL file with preference pairs
        base_model_key: Model key from configuration
        output_dir: Output directory for trained model
        config_overrides: Dictionary of config values to override
        
    Returns:
        Trained DPOTrainer_Custom instance
    """
    # Create config
    config = DPOTrainingConfig(
        base_model_key=base_model_key,
        dataset_path=dataset_path,
        output_dir=output_dir
    )
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    # Create trainer
    trainer = DPOTrainer_Custom(config)
    
    # Load model and data
    trainer.load_model_and_tokenizer()
    trainer.load_dataset(dataset_path)
    
    # Prepare and run training
    trainer.prepare_training()
    trainer.train()
    
    # Evaluate if possible
    trainer.evaluate()
    
    return trainer


if __name__ == "__main__":
    # Test DPO trainer configuration
    print("Testing DPO Trainer...")
    
    # Create test config
    config = DPOTrainingConfig(
        base_model_key='qwen2_0_5b',
        output_dir='./models/test_dpo',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        max_memory_gb=8.0,
        use_lora=True
    )
    
    print("DPO Training Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Test dataset loading (mock)
    test_preference_pairs = [
        {
            "prompt": "What is consciousness?",
            "chosen": "Consciousness is a complex phenomenon that arises from interconnected processes...",
            "rejected": "Consciousness is just brain activity.",
            "principle": "Emptiness Principle"
        },
        {
            "prompt": "How should we treat others?",
            "chosen": "We should treat others with compassion, recognizing our shared humanity...",
            "rejected": "Treat others based on what they can do for you.",
            "principle": "Boundless Care Principle"
        }
    ]
    
    # Save test data
    test_data_path = Path("./results/test_dpo_data.jsonl")
    test_data_path.parent.mkdir(exist_ok=True)
    
    with open(test_data_path, 'w') as f:
        for pair in test_preference_pairs:
            json.dump(pair, f)
            f.write('\n')
    
    print(f"Created test dataset: {test_data_path}")
    
    # Test trainer initialization (without actual training)
    try:
        trainer = DPOTrainer_Custom(config)
        print("✅ DPO Trainer initialized successfully")
        
        # Test dataset loading
        trainer.load_dataset(str(test_data_path))
        print("✅ Dataset loaded successfully")
        
    except Exception as e:
        print(f"❌ Error in DPO trainer: {e}")
    
    print("DPO Trainer test completed!")
