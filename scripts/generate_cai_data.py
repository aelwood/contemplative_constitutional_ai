#!/usr/bin/env python3
"""Generate constitutional preference pairs using a chosen constitution and model."""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def load_prompts(prompts_path: Optional[Path]) -> List[str]:
    """Load prompts from JSONL or plaintext. Falls back to baked-in prompts."""
    if prompts_path is None:
        logging.info("No prompt file supplied; using default demo prompts")
        return [
            "How should I navigate a conflict with a colleague who keeps dismissing my ideas?",
            "Is it ever acceptable to sacrifice one person's wellbeing for the greater good?",
            "I'm feeling overwhelmed by the news. What mindset can help me stay engaged without burning out?",
        ]

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=None, help="Path to prompts (JSONL with 'prompt' field or plaintext)")
    parser.add_argument("--constitution", type=Path, required=True, help="Path to constitutional markdown file")
    parser.add_argument("--model", default="qwen2_0_5b", help="Model key defined in configs/model_configs.yaml")
    parser.add_argument("--output", type=Path, default=Path("results/generated_preference_pairs.jsonl"), help="Where to write preference pairs")
    parser.add_argument("--device", default=None, help="Device override (mps, cuda, cpu)")
    parser.add_argument("--max-memory-gb", type=float, default=12.0, help="Approximate memory budget for model loading")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional cap on number of prompts")
    parser.add_argument("--principles", nargs="*", default=None, help="Optional subset of principle names to apply")
    parser.add_argument("--base-max-new-tokens", type=int, default=512, help="Token cap for generating baseline responses")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Lazily import project modules after setting up sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from cai.pipeline import CAIPipeline  # pylint: disable=import-error

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    logging.info("Loaded %d prompts", len(prompts))

    pipeline = CAIPipeline(
        constitutional_config_path=str(args.constitution),
        device=args.device,
        max_memory_gb=args.max_memory_gb,
    )

    logging.info("Loading model '%s'", args.model)
    pipeline.load_model(model_key=args.model)

    base_responses: List[str] = []
    logging.info("Generating baseline responses...")
    for prompt in prompts:
        response = pipeline.generate_text(prompt, max_new_tokens=args.base_max_new_tokens)
        base_responses.append(response)

    logging.info("Creating preference pairs with constitution: %s", args.constitution)
    preference_pairs = pipeline.create_preference_pairs(
        prompts=prompts,
        responses=base_responses,
        principle_names=args.principles,
    )

    logging.info("Saving %d preference pairs to %s", len(preference_pairs), args.output)
    pipeline.save_preference_pairs(preference_pairs, str(args.output))

    stats = pipeline.get_principle_stats(preference_pairs)
    logging.info("Principle counts: %s", stats)

    # Emit a quick sample to stdout for inspection
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


if __name__ == "__main__":
    main()