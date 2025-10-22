#!/usr/bin/env python3
"""
Contemplative Constitutional AI Evaluation Script
Configurable evaluator supporting both local and API-based models.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.contemplative_evaluator import ContemplativeEvaluator
from models.model_wrapper import ModelWrapperFactory


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Contemplative Constitutional AI models")
    parser.add_argument("--config", default="configs/evaluation_configs.yaml", help="Path to evaluation config file")
    parser.add_argument("--baseline-model", help="Baseline model key from config")
    parser.add_argument("--finetuned-model", help="Fine-tuned model key from config")
    parser.add_argument("--dataset", default="test_prompts", help="Dataset key from config")
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to evaluate")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--use-llm", action="store_true", default=True, help="Use LLM-based evaluation (default: True)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based evaluation (use rule-based)")
    parser.add_argument("--api-key", help="OpenAI API key (optional if set as environment variable)")
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM model for evaluation")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Determine evaluation method
    use_llm = args.use_llm and not args.no_llm
    
    # Initialize evaluator with evaluator model
    evaluator_model_key = None
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        if "evaluator_model" in config_data:
            evaluator_model_key = config_data["evaluator_model"].get("local_evaluator") or config_data["evaluator_model"].get("api_evaluator")
    
    evaluator = ContemplativeEvaluator(args.config, evaluator_model_key)
    
    # Load test prompts
    test_prompts = evaluator.load_test_prompts(args.dataset, args.max_prompts)
    logging.info(f"Loaded {len(test_prompts)} test prompts")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        results_dir = Path(evaluator.config["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        if args.finetuned_model:
            output_path = results_dir / f"comparison_{args.baseline_model}_vs_{args.finetuned_model}.json"
        else:
            output_path = results_dir / f"evaluation_{args.baseline_model}.json"
    
    # Run evaluation
    if args.finetuned_model:
        # Compare models
        logging.info(f"Comparing {args.baseline_model} vs {args.finetuned_model}")
        
        baseline_wrapper = ModelWrapperFactory.create_from_config_file(args.config, args.baseline_model)
        finetuned_wrapper = ModelWrapperFactory.create_from_config_file(args.config, args.finetuned_model)
        
        # Load models
        baseline_wrapper.load_model()
        finetuned_wrapper.load_model()
        
        try:
            results = evaluator.compare_models(baseline_wrapper, finetuned_wrapper, test_prompts, use_llm)
        finally:
            baseline_wrapper.unload_model()
            finetuned_wrapper.unload_model()
    else:
        # Single model evaluation
        logging.info(f"Evaluating {args.baseline_model}")
        
        model_wrapper = ModelWrapperFactory.create_from_config_file(args.config, args.baseline_model)
        model_wrapper.load_model()
        
        try:
            results = evaluator.evaluate_model(model_wrapper, test_prompts, use_llm)
        finally:
            model_wrapper.unload_model()
    
    # Save and display results
    evaluator.save_results(results, output_path)
    evaluator.print_summary(results)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
