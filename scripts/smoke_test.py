#!/usr/bin/env python3
"""
Smoke test for Contemplative Constitutional AI setup.
Verifies environment, device detection, and basic model loading.
"""

import sys
import os
import torch
import psutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.model_loader import ModelLoader, load_qwen_poc_model


def check_system_requirements():
    """Check basic system requirements and hardware."""
    print("=== System Requirements Check ===")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # Available memory
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.1f} GB")
    print(f"Available memory: {memory.available / (1024**3):.1f} GB")
    print(f"Memory usage: {memory.percent}%")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print(f"Free disk space: {disk.free / (1024**3):.1f} GB")
    
    # Check if we have enough memory for PoC
    if memory.available < 4 * (1024**3):  # 4GB minimum
        print("⚠️ Warning: Less than 4GB available memory. Close other applications.")
    else:
        print("✅ Sufficient memory available")
    
    print()


def check_pytorch_installation():
    """Check PyTorch installation and device availability."""
    print("=== PyTorch Installation Check ===")
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability (Apple Silicon)
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Metal) is available")
        if torch.backends.mps.is_built():
            print("✅ MPS backend is built")
        else:
            print("❌ MPS backend is not built")
    else:
        print("❌ MPS is not available")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA is available with {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA is not available")
    
    # Basic tensor operations
    try:
        # Test CPU tensor
        x = torch.randn(3, 3)
        print(f"✅ CPU tensor creation successful: {x.shape}")
        
        # Test MPS if available
        if torch.backends.mps.is_available():
            x_mps = x.to('mps')
            y_mps = torch.randn(3, 3, device='mps')
            z_mps = x_mps + y_mps
            print(f"✅ MPS tensor operations successful: {z_mps.shape}")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            x_cuda = x.to('cuda')
            y_cuda = torch.randn(3, 3, device='cuda')
            z_cuda = x_cuda + y_cuda
            print(f"✅ CUDA tensor operations successful: {z_cuda.shape}")
            
    except Exception as e:
        print(f"❌ Error in tensor operations: {e}")
    
    print()


def check_transformers_installation():
    """Check transformers library installation."""
    print("=== Transformers Library Check ===")
    
    try:
        from transformers import __version__ as transformers_version
        print(f"Transformers version: {transformers_version}")
        
        from transformers import AutoTokenizer
        print("✅ AutoTokenizer import successful")
        
        from transformers import AutoModelForCausalLM
        print("✅ AutoModelForCausalLM import successful")
        
    except ImportError as e:
        print(f"❌ Error importing transformers: {e}")
        return False
    
    print()
    return True


def test_model_loader():
    """Test the ModelLoader class without actually loading a model."""
    print("=== Model Loader Test ===")
    
    try:
        loader = ModelLoader()
        print("✅ ModelLoader initialization successful")
        
        # Test device detection
        device = loader.detect_device()
        print(f"✅ Device detection successful: {device}")
        
        # Test model info retrieval
        model_info = loader.get_model_info('qwen2_0_5b')
        print(f"✅ Model info retrieval successful")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Size: {model_info['model_size']}")
        print(f"   Estimated memory: {model_info['estimated_memory_gb']}GB")
        
        # Test loading config
        loading_config = loader.get_loading_config(device)
        print(f"✅ Loading config retrieved for {device}")
        
        # Test generation params
        gen_params = loader.get_generation_params('balanced')
        print(f"✅ Generation params retrieved: {len(gen_params)} parameters")
        
    except Exception as e:
        print(f"❌ Error in ModelLoader test: {e}")
        return False
    
    print()
    return True


def test_actual_model_loading():
    """Test actual model loading (optional - requires network and time)."""
    print("=== Actual Model Loading Test ===")
    
    response = input("Load actual QWEN2-0.5B model? This requires ~1GB download and 2-4GB RAM. (y/N): ")
    if response.lower() != 'y':
        print("⏭️ Skipping actual model loading")
        return True
    
    try:
        print("Loading QWEN2-0.5B model...")
        model, tokenizer = load_qwen_poc_model()
        
        print("✅ Model and tokenizer loaded successfully")
        print(f"   Model device: {model.device}")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        
        # Test basic generation
        print("Testing basic generation...")
        prompt = "What is the meaning of life?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to same device as model
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation successful")
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response[len(prompt):].strip()}")
        
        # Clean up memory
        del model, tokenizer, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error in model loading test: {e}")
        return False
    
    print()
    return True


def main():
    """Run all smoke tests."""
    print("🔬 Contemplative Constitutional AI - Smoke Test")
    print("=" * 50)
    
    tests = [
        check_system_requirements,
        check_pytorch_installation,
        check_transformers_installation,
        test_model_loader,
        test_actual_model_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("=" * 50)
    print("🔬 Smoke Test Results")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Ready for PoC development.")
    elif passed >= total - 1:
        print("⚠️ Most tests passed. Check warnings above.")
    else:
        print("❌ Multiple tests failed. Check errors above.")
    
    # Memory usage after tests
    memory = psutil.virtual_memory()
    print(f"\nMemory usage after tests: {memory.percent}% ({memory.available / (1024**3):.1f} GB available)")


if __name__ == "__main__":
    main()
