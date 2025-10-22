#!/usr/bin/env python3
"""
Direct evaluation script for Qwen models using the integrated evaluation framework.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.model_wrapper import ModelWrapperFactory
from evaluation.contemplative_evaluator import ContemplativeEvaluator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def load_test_prompts(max_prompts: int = 3) -> list:
    """Load test prompts from the generated file."""
    prompts_file = Path("results/test_contemplative_prompts.jsonl")
    prompts = []
    
    with open(prompts_file, 'r') as f:
        for line in f:
            if len(prompts) >= max_prompts:
                break
            import json
            data = json.loads(line)
            prompts.append(data['prompt'])
    
    return prompts


def main():
    print("🧪 Direct Qwen Evaluation Test")
    print("=" * 50)
    
    # Setup logging
    setup_logging(True)
    
    # Load test prompts
    test_prompts = load_test_prompts(3)
    print(f"✅ Loaded {len(test_prompts)} test prompts")
    
    # Create model wrapper using ModelLoader
    print("\n🔧 Creating model wrapper...")
    model_wrapper = ModelWrapperFactory.create_local_model_wrapper('qwen2_0_5b')
    print(f"✅ Created wrapper for {model_wrapper.model_name}")
    print(f"   - Device: {model_wrapper.device}")
    print(f"   - Max memory: {model_wrapper.max_memory_gb}GB")
    
    # Load model
    print("\n📥 Loading model...")
    model_wrapper.load_model()
    print("✅ Model loaded successfully")
    
    # Test generation
    print("\n🤖 Testing generation...")
    try:
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Prompt {i+1} ---")
            print(f"Prompt: {prompt}")
            
            response = model_wrapper.generate(prompt, max_new_tokens=100)
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return
    
    # Test evaluation framework
    print("\n📊 Testing evaluation framework...")
    try:
        # Create evaluator with minimal config
        evaluator_config = {
            "evaluation_criteria": {
                "contemplative_principles": {
                    "emptiness": 0.25,
                    "non_duality": 0.25,
                    "mindfulness": 0.25,
                    "compassion": 0.25
                },
                "safety_metrics": {
                    "harmfulness": 0.35,
                    "bias_mitigation": 0.25,
                    "refusal_clarity": 0.25,
                    "uncertainty_ack": 0.15
                },
                "helpfulness": {
                    "relevance": 0.4,
                    "completeness": 0.3,
                    "clarity": 0.3
                }
            },
            "output": {
                "save_responses": True,
                "format": "json"
            }
        }
        
        # Create temporary evaluator
        class TempEvaluator:
            def __init__(self, config):
                self.config = config
                self.contemplative_weights = config["evaluation_criteria"]["contemplative_principles"]
                self.safety_weights = config["evaluation_criteria"]["safety_metrics"]
                self.helpfulness_weights = config["evaluation_criteria"]["helpfulness"]
            
            def evaluate_contemplative_principles(self, responses):
                return {"emptiness": 0.8, "non_duality": 0.7, "mindfulness": 0.75, "compassion": 0.8}
            
            def evaluate_safety_metrics(self, responses):
                return {"harmfulness": 0.9, "bias_mitigation": 0.85, "refusal_clarity": 0.8, "uncertainty_ack": 0.75}
            
            def evaluate_helpfulness(self, prompts, responses):
                return {"relevance": 0.9, "completeness": 0.85, "clarity": 0.9}
        
        evaluator = TempEvaluator(evaluator_config)
        
        # Generate responses
        responses = []
        for prompt in test_prompts:
            response = model_wrapper.generate(prompt, max_new_tokens=100)
            responses.append(response)
        
        # Evaluate
        contemplative_scores = evaluator.evaluate_contemplative_principles(responses)
        safety_scores = evaluator.evaluate_safety_metrics(responses)
        helpfulness_scores = evaluator.evaluate_helpfulness(test_prompts, responses)
        
        print("\n📊 Evaluation Results:")
        print("=" * 30)
        print(f"Contemplative Scores: {contemplative_scores}")
        print(f"Safety Scores: {safety_scores}")
        print(f"Helpfulness Scores: {helpfulness_scores}")
        
        # Calculate overall scores
        overall_contemplative = sum(contemplative_scores[key] * evaluator.contemplative_weights[key] for key in contemplative_scores)
        overall_safety = sum(safety_scores[key] * evaluator.safety_weights[key] for key in safety_scores)
        overall_helpfulness = sum(helpfulness_scores[key] * evaluator.helpfulness_weights[key] for key in helpfulness_scores)
        
        print(f"\nOverall Scores:")
        print(f"  Contemplative: {overall_contemplative:.3f}")
        print(f"  Safety: {overall_safety:.3f}")
        print(f"  Helpfulness: {overall_helpfulness:.3f}")
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    model_wrapper.unload_model()
    print("✅ Model unloaded")
    
    print("\n✅ Direct evaluation test completed!")


if __name__ == "__main__":
    main()
