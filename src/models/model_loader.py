"""
Model loading utilities for Contemplative Constitutional AI.
Optimized for Apple Silicon (MPS) and CUDA devices.
"""

import torch
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and configuration of language models across different hardware."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ModelLoader with configuration.
        
        Args:
            config_path: Path to model configuration YAML file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "model_configs.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def detect_device(self) -> str:
        """
        Detect the best available device for model loading.
        
        Returns:
            Device string: 'mps', 'cuda', or 'cpu'
        """
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) detected and available")
            return 'mps'
        elif torch.cuda.is_available():
            logger.info(f"CUDA detected with {torch.cuda.device_count()} GPU(s)")
            return 'cuda'
        else:
            logger.info("No GPU acceleration available, using CPU")
            return 'cpu'
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """
        Get model information from configuration.
        
        Args:
            model_key: Key for the model in configuration
            
        Returns:
            Dictionary with model information
        """
        # Search across all model categories
        for category in ['poc_models', 'development_models', 'production_models']:
            if model_key in self.config.get(category, {}):
                return self.config[category][model_key]
        
        raise ValueError(f"Model '{model_key}' not found in configuration")
    
    def get_loading_config(self, device: str, quantization: Optional[str] = None) -> Dict[str, Any]:
        """
        Get loading configuration based on device and quantization options.
        
        Args:
            device: Target device ('mps', 'cuda', 'cpu')
            quantization: Optional quantization ('int8', 'int4')
            
        Returns:
            Loading configuration dictionary
        """
        # Base loading config
        if device == 'mps':
            base_config = self.config['loading_configs']['macbook_m2'].copy()
        elif device == 'cuda':
            base_config = self.config['loading_configs']['single_gpu'].copy()
        else:
            base_config = self.config['loading_configs']['single_gpu'].copy()
            base_config['device_map'] = 'cpu'
        
        # Add quantization if specified
        if quantization and quantization in self.config['quantization']:
            quant_config = self.config['quantization'][quantization]
            base_config.update(quant_config)
        
        return base_config
    
    def create_quantization_config(self, quantization_type: str) -> Optional[BitsAndBytesConfig]:
        """
        Create BitsAndBytesConfig for quantization.
        
        Args:
            quantization_type: Type of quantization ('int8' or 'int4')
            
        Returns:
            BitsAndBytesConfig object or None
        """
        if quantization_type not in self.config['quantization']:
            return None
        
        quant_config = self.config['quantization'][quantization_type]
        
        if quantization_type == 'int8':
            return BitsAndBytesConfig(
                load_in_8bit=quant_config['load_in_8bit'],
                llm_int8_threshold=quant_config['llm_int8_threshold'],
                llm_int8_enable_fp32_cpu_offload=quant_config['llm_int8_enable_fp32_cpu_offload']
            )
        elif quantization_type == 'int4':
            return BitsAndBytesConfig(
                load_in_4bit=quant_config['load_in_4bit'],
                bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type']
            )
        
        return None
    
    def load_model_and_tokenizer(
        self,
        model_key: str,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        max_memory_gb: Optional[float] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer with optimal configuration.
        
        Args:
            model_key: Key for the model in configuration
            device: Target device (auto-detected if None)
            quantization: Quantization type ('int8', 'int4', or None)
            max_memory_gb: Maximum memory to use in GB
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Get model info
        model_info = self.get_model_info(model_key)
        model_name = model_info['model_name']
        
        # Detect device if not specified
        if device is None:
            device = self.detect_device()
        
        # Get loading configuration
        loading_config = self.get_loading_config(device, quantization)
        
        # Handle torch_dtype
        if 'torch_dtype' in loading_config:
            loading_config['torch_dtype'] = getattr(torch, loading_config['torch_dtype'])
        
        # Create quantization config if needed
        quantization_config = None
        if quantization:
            quantization_config = self.create_quantization_config(quantization)
            if quantization_config:
                loading_config['quantization_config'] = quantization_config
        
        # Handle max memory constraint
        if max_memory_gb and device == 'mps':
            # For MPS, we can't directly control memory allocation
            # but we can use smaller precision and enable optimizations
            loading_config['torch_dtype'] = torch.float16
            logger.info(f"Using FP16 for memory efficiency (limit: {max_memory_gb}GB)")
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Target device: {device}")
        logger.info(f"Estimated memory: {model_info.get('estimated_memory_gb', 'unknown')}GB")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=loading_config.get('trust_remote_code', True)
            )
            
            # Ensure pad token exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **loading_config
            )
            
            # Move to device if needed (for non-auto device mapping)
            if device == 'mps' and hasattr(model, 'to'):
                model = model.to('mps')
            
            logger.info(f"Model loaded successfully on {device}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            
            # Fallback options
            if device == 'mps' and not quantization:
                logger.info("Retrying with CPU fallback...")
                return self.load_model_and_tokenizer(
                    model_key, device='cpu', quantization=quantization
                )
            elif not quantization and device != 'cpu':
                logger.info("Retrying with 8-bit quantization...")
                return self.load_model_and_tokenizer(
                    model_key, device=device, quantization='int8'
                )
            else:
                raise e
    
    def get_generation_params(self, style: str = 'balanced') -> Dict[str, Any]:
        """
        Get generation parameters for different styles.
        
        Args:
            style: Generation style ('conservative', 'balanced', 'creative')
            
        Returns:
            Generation parameters dictionary
        """
        if style not in self.config['generation_params']:
            logger.warning(f"Unknown generation style '{style}', using 'balanced'")
            style = 'balanced'
        
        return self.config['generation_params'][style].copy()


def load_qwen_poc_model(max_memory_gb: float = 12.0) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Convenience function to load QWEN2-0.5B for PoC on MacBook M2.
    
    Args:
        max_memory_gb: Maximum memory to use in GB
        
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader()
    return loader.load_model_and_tokenizer(
        model_key='qwen2_0_5b',
        max_memory_gb=max_memory_gb
    )


if __name__ == "__main__":
    # Basic smoke test
    print("Testing model loader...")
    
    loader = ModelLoader()
    device = loader.detect_device()
    print(f"Detected device: {device}")
    
    # Test model info retrieval
    try:
        model_info = loader.get_model_info('qwen2_0_5b')
        print(f"PoC model info: {model_info}")
    except Exception as e:
        print(f"Error getting model info: {e}")
