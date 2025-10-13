#!/usr/bin/env python3
"""Train a DPO adapter on constitutional preference pairs."""

import argparse
import logging
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to preference-pair JSONL")
    parser.add_argument("--base-model", default="qwen2_0_5b", help="Model key from configs/model_configs.yaml")
    parser.add_argument("--output", type=Path, default=Path("models/contemplative_dpo"), help="Directory to store fine-tuned weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--device", default=None, help="Force device (mps, cuda, cpu)")
    parser.add_argument("--max-memory-gb", type=float, default=12.0, help="Approximate memory budget for model loading")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Fraction of data reserved for evaluation")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA adapters and finetune full model")
    parser.add_argument("--logging-steps", type=int, default=10, help="Training logging interval")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--load-best-model", action="store_true", help="Enable loading best checkpoint at end (disabled by default)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from training.dpo_trainer import (  # pylint: disable=import-error
        DPOTrainingConfig,
        DPOTrainer_Custom,
    )

    config = DPOTrainingConfig(
        base_model_key=args.base_model,
        dataset_path=str(args.dataset),
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        beta=args.beta,
        device=args.device,
        max_memory_gb=args.max_memory_gb,
        use_lora=not args.no_lora,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model,
    )

    trainer = DPOTrainer_Custom(config)

    logging.info("Loading base model: %s", args.base_model)
    trainer.load_model_and_tokenizer()

    logging.info("Loading dataset: %s", args.dataset)
    trainer.load_dataset(str(args.dataset), eval_split=args.eval_split)

    logging.info("Preparing training components")
    trainer.prepare_training()

    logging.info("Starting training run")
    trainer.train()

    if not args.no_eval:
        logging.info("Running evaluation on held-out split")
        trainer.evaluate()

    logging.info("Training complete. Artifacts stored in %s", args.output)


if __name__ == "__main__":
    main()
