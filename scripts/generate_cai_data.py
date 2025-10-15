#!/usr/bin/env python3
"""Generate constitutional preference pairs using a chosen constitution and model."""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def load_prompts_from_file(prompts_path: Path) -> List[str]:
    """Load prompts from JSONL or plaintext file."""
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

    if prompts_path.suffix.lower() == ".jsonl":
        prompts: List[str] = []
        with prompts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prompt = data.get("prompt")
                if not prompt:
                    raise ValueError("JSONL entries must contain a 'prompt' field")
                prompts.append(prompt)
        if not prompts:
            raise ValueError("No prompts found in JSONL file")
        return prompts

    # Treat everything else as plaintext with one prompt per line
    with prompts_path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError("No prompts found in text file")
    return prompts


def load_prompts(
    prompts_path: Optional[Path],
    use_ailuminate: bool = False,
    ailuminate_path: Optional[Path] = None,
    hazard_categories: Optional[List[str]] = None,
    persona_types: Optional[List[str]] = None,
    max_prompts: Optional[int] = None
) -> tuple[List[str], List[Dict]]:
    """
    Load prompts from file or AILuminate dataset.
    
    Returns:
        Tuple of (prompts, metadata_list) where metadata contains prompt IDs for splitting
    """
    if use_ailuminate:
        from data.ailuminate_loader import AILuminateLoader
        
        # Default to demo dataset
        if ailuminate_path is None:
            ailuminate_path = Path("data/benchmarks/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv")
        
        if not ailuminate_path.exists():
            raise FileNotFoundError(
                f"AILuminate dataset not found: {ailuminate_path}\n"
                "Run: git submodule update --init --recursive"
            )
        
        logging.info(f"Loading AILuminate dataset from: {ailuminate_path}")
        loader = AILuminateLoader(str(ailuminate_path))
        
        # Print statistics
        loader.print_statistics()
        
        # Get filtered prompts
        prompt_dicts = loader.get_prompts(
            hazard_categories=hazard_categories,
            persona_types=persona_types,
            n_samples=max_prompts,
            random_state=42
        )
        
        prompts = [p['prompt'] for p in prompt_dicts]
        metadata = [{'prompt_id': p['prompt_id'], 'hazard': p['hazard_category']} 
                   for p in prompt_dicts]
        
        logging.info(f"Loaded {len(prompts)} AILuminate prompts")
        return prompts, metadata
    
    # Load from file or use defaults
    if prompts_path is None:
        logging.info("No prompt file supplied; using default demo prompts")
        prompts = [
            "How should I navigate a conflict with a colleague who keeps dismissing my ideas?",
            "Is it ever acceptable to sacrifice one person's wellbeing for the greater good?",
            "I'm feeling overwhelmed by the news. What mindset can help me stay engaged without burning out?",
        ]
    else:
        prompts = load_prompts_from_file(prompts_path)
    
    # Create simple metadata (hash-based IDs)
    import hashlib
    metadata = [{'prompt_id': hashlib.md5(p.encode()).hexdigest()[:16]} 
               for p in prompts]
    
    return prompts, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Prompt source arguments
    parser.add_argument("--prompts", type=Path, default=None, help="Path to prompts (JSONL with 'prompt' field or plaintext)")
    parser.add_argument("--use-ailuminate", action="store_true", help="Load prompts from AILuminate dataset")
    parser.add_argument("--ailuminate-path", type=Path, default=None, 
                       help="Path to AILuminate CSV (default: data/benchmarks/ailuminate/...)")
    parser.add_argument("--hazard-categories", nargs="*", default=None,
                       help="Filter AILuminate by hazard categories (e.g., vcr cse hte)")
    parser.add_argument("--persona-types", nargs="*", default=None,
                       help="Filter AILuminate by persona types (e.g., normal skilled)")
    
    # Model and constitution arguments
    parser.add_argument("--constitution", type=Path, required=True, help="Path to constitutional markdown file")
    parser.add_argument("--model", default="qwen2_0_5b", help="Model key defined in configs/model_configs.yaml")
    parser.add_argument("--device", default=None, help="Device override (mps, cuda, cpu)")
    parser.add_argument("--max-memory-gb", type=float, default=12.0, help="Approximate memory budget for model loading")
    
    # Output arguments
    parser.add_argument("--output", type=Path, default=Path("results/generated_preference_pairs.jsonl"), 
                       help="Where to write preference pairs")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional cap on number of prompts")
    parser.add_argument("--principles", nargs="*", default=None, help="Optional subset of principle names to apply")
    parser.add_argument("--base-max-new-tokens", type=int, default=512, help="Token cap for generating baseline responses")
    
    # Train/test split arguments
    parser.add_argument("--create-split", action="store_true", help="Create train/test split and save configuration")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction of data for test set (default: 0.1)")
    parser.add_argument("--split-config", type=Path, default=Path("data/splits/default_split.json"),
                       help="Path to split configuration file")
    parser.add_argument("--split-only", choices=['train', 'test'], default=None,
                       help="Generate only train or test split (requires existing split config)")
    
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    from cai.pipeline import CAIPipeline  # pylint: disable=import-error
    from data.split_manager import SplitManager

    # Load prompts
    prompts, metadata = load_prompts(
        prompts_path=args.prompts,
        use_ailuminate=args.use_ailuminate,
        ailuminate_path=args.ailuminate_path,
        hazard_categories=args.hazard_categories,
        persona_types=args.persona_types,
        max_prompts=args.max_prompts
    )

    logging.info("Loaded %d prompts", len(prompts))

    # Handle train/test splitting
    split_manager = None
    if args.create_split:
        # Create new split
        logging.info("Creating train/test split...")
        split_manager = SplitManager(str(args.split_config))
        prompt_ids = [m['prompt_id'] for m in metadata]
        train_ids, test_ids = split_manager.create_split(
            prompt_ids=prompt_ids,
            test_size=args.test_size,
            random_state=42,
            metadata={
                'dataset': 'ailuminate' if args.use_ailuminate else 'custom',
                'n_prompts': len(prompts),
                'hazard_categories': args.hazard_categories,
                'persona_types': args.persona_types,
            }
        )
        split_manager.print_statistics()
    
    if args.split_only:
        # Use existing split configuration
        if not args.split_config.exists():
            raise FileNotFoundError(
                f"Split configuration not found: {args.split_config}\n"
                "Run with --create-split first to create the split"
            )
        
        split_manager = SplitManager(str(args.split_config))
        split_manager.print_statistics()
        
        # Filter prompts/metadata to requested split
        if args.split_only == 'train':
            filtered_metadata = split_manager.filter_train(metadata)
        else:
            filtered_metadata = split_manager.filter_test(metadata)
        
        # Update prompts list to match filtered metadata
        filtered_indices = [i for i, m in enumerate(metadata) if m in filtered_metadata]
        prompts = [prompts[i] for i in filtered_indices]
        metadata = filtered_metadata
        
        logging.info(f"Filtered to {args.split_only} split: {len(prompts)} prompts")

    # Initialize CAI pipeline
    pipeline = CAIPipeline(
        constitutional_config_path=str(args.constitution),
        device=args.device,
        max_memory_gb=args.max_memory_gb,
    )

    logging.info("Loading model '%s'", args.model)
    pipeline.load_model(model_key=args.model)

    # Generate baseline responses
    base_responses: List[str] = []
    logging.info("Generating baseline responses...")
    for prompt in prompts:
        response = pipeline.generate_text(prompt, max_new_tokens=args.base_max_new_tokens)
        base_responses.append(response)

    # Create preference pairs
    logging.info("Creating preference pairs with constitution: %s", args.constitution)
    preference_pairs = pipeline.create_preference_pairs(
        prompts=prompts,
        responses=base_responses,
        principle_names=args.principles,
    )

    # Add prompt_id to preference pairs for split tracking
    for pair, meta in zip(preference_pairs, metadata * 4):  # 4 principles per prompt
        if not hasattr(pair, 'metadata'):
            pair.metadata = {}
        pair.metadata['prompt_id'] = meta['prompt_id']

    # Save preference pairs
    logging.info("Saving %d preference pairs to %s", len(preference_pairs), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_preference_pairs(preference_pairs, str(args.output))

    # Print statistics
    stats = pipeline.get_principle_stats(preference_pairs)
    logging.info("Principle counts: %s", stats)

    # Print sample
    if preference_pairs:
        sample = preference_pairs[0]
        preview = {
            "prompt": sample.prompt,
            "principle": sample.principle,
            "rejected_excerpt": sample.rejected[:120],
            "chosen_excerpt": sample.chosen[:120],
        }
        print("\nSample pair:")
        print(json.dumps(preview, indent=2, ensure_ascii=False))
    
    # Print split info if applicable
    if args.split_only:
        print(f"\n✓ Generated {args.split_only} split: {len(preference_pairs)} preference pairs")
        print(f"  Split configuration: {args.split_config}")
    elif args.create_split:
        print(f"\n✓ Created train/test split configuration: {args.split_config}")
        print(f"  Train: {len(split_manager.get_split()[0])} prompts")
        print(f"  Test: {len(split_manager.get_split()[1])} prompts")


if __name__ == "__main__":
    main()