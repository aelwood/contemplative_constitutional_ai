"""
Flexible model wrapper for evaluation.
Supports both local models (transformers) and API-based models.
Integrates with existing ModelLoader for consistency.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .model_loader import ModelLoader


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "unknown")
        self.model_type = config.get("model_type", "unknown")
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from model."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        pass


class LocalModelWrapper(ModelWrapper):
    """Wrapper for local models using transformers and existing ModelLoader."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = config.get("device", "mps")
        self.max_memory_gb = config.get("max_memory_gb", 4.0)
        self.quantization = config.get("quantization", None)
        self.model_loader = ModelLoader()
        
    def load_model(self) -> None:
        """Load local model and tokenizer using existing ModelLoader."""
        self.logger.info(f"Loading local model: {self.model_name}")
        
        # Use existing ModelLoader for consistency
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(
            model_key=self.model_name,
            device=self.device,
            quantization=self.quantization,
            max_memory_gb=self.max_memory_gb
        )
        
        self.logger.info(f"Model loaded successfully on {self.device}")
    
    def load_finetuned_model(self, adapter_path: str) -> None:
        """Load fine-tuned model with LoRA adapters."""
        if self.model is None:
            self.load_model()
        
        self.logger.info(f"Loading fine-tuned model from: {adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            device_map=self.device,
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        """Generate response from local model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get("do_sample", True),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        self.logger.info("Model unloaded and memory freed")


class AnthropicModelWrapper(ModelWrapper):
    """Wrapper for Anthropic Claude models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not self.api_key:
            raise ValueError(f"API key not found in environment variable: {config.get('api_key_env')}")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def load_model(self) -> None:
        """No loading needed for API models."""
        self.logger.info(f"Anthropic client initialized for {self.model_name}")
    
    def generate(self, prompt: str, max_new_tokens: int = 1000, **kwargs) -> str:
        """Generate response using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_new_tokens,
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return f"Error: {str(e)}"
    
    def unload_model(self) -> None:
        """No unloading needed for API models."""
        pass


class OpenAIModelWrapper(ModelWrapper):
    """Wrapper for OpenAI models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError(f"API key not found in environment variable: {config.get('api_key_env')}")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def load_model(self) -> None:
        """No loading needed for API models."""
        self.logger.info(f"OpenAI client initialized for {self.model_name}")
    
    def generate(self, prompt: str, max_new_tokens: int = 1000, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_new_tokens,
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"
    
    def unload_model(self) -> None:
        """No unloading needed for API models."""
        pass


class GoogleModelWrapper(ModelWrapper):
    """Wrapper for Google Gemini models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv(config.get("api_key_env", "GOOGLE_API_KEY"))
        if not self.api_key:
            raise ValueError(f"API key not found in environment variable: {config.get('api_key_env')}")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def load_model(self) -> None:
        """No loading needed for API models."""
        self.logger.info(f"Google Gemini client initialized for {self.model_name}")
    
    def generate(self, prompt: str, max_new_tokens: int = 1000, **kwargs) -> str:
        """Generate response using Google Gemini API."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=kwargs.get("temperature", 0.7),
                )
            )
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"Google Gemini API error: {e}")
            return f"Error: {str(e)}"
    
    def unload_model(self) -> None:
        """No unloading needed for API models."""
        pass


class ModelWrapperFactory:
    """Factory for creating model wrappers."""
    
    @staticmethod
    def create_wrapper(config: Dict[str, Any]) -> ModelWrapper:
        """Create appropriate model wrapper based on configuration."""
        model_type = config.get("model_type", "local")
        
        if model_type == "local":
            return LocalModelWrapper(config)
        elif model_type == "anthropic":
            return AnthropicModelWrapper(config)
        elif model_type == "openai":
            return OpenAIModelWrapper(config)
        elif model_type == "google":
            return GoogleModelWrapper(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_from_config_file(config_path: str, model_key: str) -> ModelWrapper:
        """Create wrapper from configuration file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Handle both evaluation configs and model configs
        if "models" in config_data:
            # Evaluation config format - check if it's an API model
            if model_key in config_data["models"]:
                model_config = config_data["models"][model_key]
            else:
                # Try to find in api_models
                if "api_models" in config_data["models"] and model_key in config_data["models"]["api_models"]:
                    model_config = config_data["models"]["api_models"][model_key]
                else:
                    raise ValueError(f"Model '{model_key}' not found in evaluation config")
        else:
            # Model config format - use existing ModelLoader
            model_loader = ModelLoader(config_path)
            model_info = model_loader.get_model_info(model_key)
            model_config = {
                "model_name": model_key,
                "model_type": "local",
                "device": model_loader.detect_device(),
                "max_memory_gb": model_info.get("estimated_memory_gb", 4.0)
            }
        
        return ModelWrapperFactory.create_wrapper(model_config)
    
    @staticmethod
    def create_local_model_wrapper(model_key: str, device: Optional[str] = None, 
                                  max_memory_gb: Optional[float] = None) -> LocalModelWrapper:
        """Create local model wrapper using existing ModelLoader configuration."""
        model_loader = ModelLoader()
        model_info = model_loader.get_model_info(model_key)
        
        config = {
            "model_name": model_key,
            "model_type": "local",
            "device": device or model_loader.detect_device(),
            "max_memory_gb": max_memory_gb or model_info.get("estimated_memory_gb", 4.0),
            "quantization": None
        }
        
        return LocalModelWrapper(config)
    
    @staticmethod
    def create_from_model_loader(model_key: str, device: Optional[str] = None, 
                               max_memory_gb: Optional[float] = None) -> LocalModelWrapper:
        """Create local model wrapper using existing ModelLoader configuration."""
        model_loader = ModelLoader()
        model_info = model_loader.get_model_info(model_key)
        
        config = {
            "model_name": model_key,
            "model_type": "local",
            "device": device or model_loader.detect_device(),
            "max_memory_gb": max_memory_gb or model_info.get("estimated_memory_gb", 4.0),
            "quantization": None
        }
        
        return LocalModelWrapper(config)
