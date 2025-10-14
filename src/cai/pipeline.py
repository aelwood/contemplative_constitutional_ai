"""
Constitutional AI Pipeline for Contemplative Constitutional AI.
Handles the end-to-end process of generating critiques and revisions.
"""

import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional.config_parser import ConstitutionalParser, ConstitutionalPrinciple
from models.model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CAIExample:
    """Represents a complete Constitutional AI example with all stages."""
    original_prompt: str
    original_response: str
    principle_name: str
    critique: str
    revised_response: str
    metadata: Dict[str, Any]


@dataclass
class PreferencePair:
    """Represents a preference pair for DPO training."""
    prompt: str
    chosen: str  # revised response
    rejected: str  # original response
    principle: str
    metadata: Dict[str, Any]


class CAIPipeline:
    """Constitutional AI pipeline for generating critiques and revisions."""
    
    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        constitutional_config_path: Optional[str] = None,
        device: Optional[str] = None,
        max_memory_gb: float = 12.0
    ):
        """
        Initialize the CAI pipeline.
        
        Args:
            model_loader: Pre-configured ModelLoader instance
            constitutional_config_path: Path to constitutional principles markdown
            device: Target device ('mps', 'cuda', 'cpu', or None for auto-detect)
            max_memory_gb: Maximum memory to use in GB
        """
        # Set up model loader
        if model_loader is None:
            model_loader = ModelLoader()
        self.model_loader = model_loader
        
        # Set up constitutional parser
        self.constitutional_parser = ConstitutionalParser()
        if constitutional_config_path is None:
            constitutional_config_path = str(
                Path(__file__).parent.parent.parent
                / "data"
                / "constitutions"
                / "contemplative-constitution-1.md"
            )
        
        logger.info(f"Loading constitutional principles from: {constitutional_config_path}")
        self.principles = self.constitutional_parser.parse_markdown_principles(constitutional_config_path)
        logger.info(f"Loaded {len(self.principles)} constitutional principles")
        
        # Model and tokenizer will be loaded on demand
        self.model = None
        self.tokenizer = None
        self.device = device
        self.max_memory_gb = max_memory_gb
        
        # Generation parameters
        self.generation_params = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
            'pad_token_id': None  # Will be set when tokenizer is loaded
        }
    
    def load_model(self, model_key: str = 'qwen2_0_5b', force_reload: bool = False) -> None:
        """
        Load the model and tokenizer for generation.
        
        Args:
            model_key: Model key from configuration
            force_reload: Force reloading even if model is already loaded
        """
        if self.model is not None and not force_reload:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading model: {model_key}")
        try:
            self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(
                model_key=model_key,
                device=self.device,
                max_memory_gb=self.max_memory_gb
            )
            
            # Update generation parameters
            self.generation_params['pad_token_id'] = self.tokenizer.eos_token_id
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to mock generation for testing")
            self.model = None
            self.tokenizer = None
    
    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override default max_new_tokens
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            # Mock generation for testing
            logger.warning("Using mock generation - no model loaded")
            return f"[MOCK] Generated response for: {prompt[:50]}..."
        
        # Prepare generation parameters
        gen_params = self.generation_params.copy()
        if max_new_tokens is not None:
            gen_params['max_new_tokens'] = max_new_tokens
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to model device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_params)
            
            # Decode and extract new text
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = full_response[len(prompt):].strip()
            
            return new_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[ERROR] Generation failed: {str(e)}"
    
    def generate_critique(
        self, 
        original_prompt: str, 
        original_response: str, 
        principle: ConstitutionalPrinciple,
        context: str = ""
    ) -> str:
        """
        Generate a critique for a response using a constitutional principle.
        
        Args:
            original_prompt: The original user prompt
            original_response: The model's original response
            principle: Constitutional principle to use for critique
            context: Additional context for the critique
            
        Returns:
            Generated critique
        """
        # Create critique prompt using the principle
        critique_prompt = principle.create_critique_prompt(original_response, context)
        
        # Generate critique
        critique = self.generate_text(critique_prompt, max_new_tokens=256)
        
        logger.debug(f"Generated critique for principle '{principle.name}'")
        return critique
    
    def generate_revision(
        self,
        original_prompt: str,
        original_response: str,
        critique: str,
        principle: ConstitutionalPrinciple,
        context: str = ""
    ) -> str:
        """
        Generate a revised response using the critique and constitutional principle.
        
        Args:
            original_prompt: The original user prompt
            original_response: The model's original response
            critique: The critique of the original response
            principle: Constitutional principle to use for revision
            context: Additional context for the revision
            
        Returns:
            Generated revised response
        """
        # Create revision prompt using the principle
        revision_prompt = principle.create_revision_prompt(original_response, critique, context)
        
        # Generate revision
        revised_response = self.generate_text(revision_prompt, max_new_tokens=512)
        
        logger.debug(f"Generated revision for principle '{principle.name}'")
        return revised_response
    
    def process_single_example(
        self,
        prompt: str,
        response: str,
        principle_names: Optional[List[str]] = None,
        context: str = ""
    ) -> List[CAIExample]:
        """
        Process a single prompt-response pair through the full CAI pipeline.
        
        Args:
            prompt: Original user prompt
            response: Original model response
            principle_names: List of principle names to apply (all if None)
            context: Additional context
            
        Returns:
            List of CAIExample objects, one per principle
        """
        if principle_names is None:
            principles_to_use = self.principles
        else:
            principles_to_use = []
            for name in principle_names:
                principle = self.constitutional_parser.get_principle_by_name(name)
                if principle:
                    principles_to_use.append(principle)
                else:
                    logger.warning(f"Principle '{name}' not found")
        
        examples = []
        
        for principle in principles_to_use:
            try:
                # Generate critique
                critique = self.generate_critique(prompt, response, principle, context)
                
                # Generate revision
                revised_response = self.generate_revision(prompt, response, critique, principle, context)
                
                # Create CAI example
                example = CAIExample(
                    original_prompt=prompt,
                    original_response=response,
                    principle_name=principle.name,
                    critique=critique,
                    revised_response=revised_response,
                    metadata={
                        'context': context,
                        'principle_critique_template': principle.critique_template,
                        'principle_revision_guideline': principle.revision_guideline
                    }
                )
                
                examples.append(example)
                logger.debug(f"Processed example with principle '{principle.name}'")
                
            except Exception as e:
                logger.error(f"Failed to process example with principle '{principle.name}': {e}")
        
        return examples
    
    def create_preference_pairs(
        self,
        prompts: List[str],
        responses: List[str],
        principle_names: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> List[PreferencePair]:
        """
        Create preference pairs from prompts and responses using constitutional AI.
        
        Args:
            prompts: List of user prompts
            responses: List of corresponding model responses
            principle_names: List of principle names to apply (all if None)
            show_progress: Whether to show progress bar
            
        Returns:
            List of PreferencePair objects
        """
        if len(prompts) != len(responses):
            raise ValueError("Prompts and responses must have the same length")
        
        preference_pairs = []
        
        iterator = zip(prompts, responses)
        if show_progress:
            iterator = tqdm(iterator, total=len(prompts), desc="Creating preference pairs")
        
        for prompt, response in iterator:
            # Process through CAI pipeline
            cai_examples = self.process_single_example(prompt, response, principle_names)
            
            # Convert to preference pairs
            for example in cai_examples:
                preference_pair = PreferencePair(
                    prompt=example.original_prompt,
                    chosen=example.revised_response,  # Revised is preferred
                    rejected=example.original_response,  # Original is rejected
                    principle=example.principle_name,
                    metadata={
                        'critique': example.critique,
                        'cai_metadata': example.metadata
                    }
                )
                preference_pairs.append(preference_pair)
        
        logger.info(f"Created {len(preference_pairs)} preference pairs")
        return preference_pairs
    
    def save_preference_pairs(self, preference_pairs: List[PreferencePair], output_path: str) -> None:
        """
        Save preference pairs to a JSONL file.
        
        Args:
            preference_pairs: List of preference pairs to save
            output_path: Path to output JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in preference_pairs:
                json.dump(asdict(pair), f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(preference_pairs)} preference pairs to {output_path}")
    
    def load_preference_pairs(self, input_path: str) -> List[PreferencePair]:
        """
        Load preference pairs from a JSONL file.
        
        Args:
            input_path: Path to input JSONL file
            
        Returns:
            List of PreferencePair objects
        """
        preference_pairs = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    pair = PreferencePair(**data)
                    preference_pairs.append(pair)
        
        logger.info(f"Loaded {len(preference_pairs)} preference pairs from {input_path}")
        return preference_pairs
    
    def get_principle_stats(self, preference_pairs: List[PreferencePair]) -> Dict[str, int]:
        """
        Get statistics about principle usage in preference pairs.
        
        Args:
            preference_pairs: List of preference pairs
            
        Returns:
            Dictionary mapping principle names to counts
        """
        stats = {}
        for pair in preference_pairs:
            stats[pair.principle] = stats.get(pair.principle, 0) + 1
        return stats


def create_cai_pipeline(
    model_key: str = 'qwen2_0_5b',
    constitutional_config_path: Optional[str] = None,
    device: Optional[str] = None
) -> CAIPipeline:
    """
    Convenience function to create and configure a CAI pipeline.
    
    Args:
        model_key: Model key from configuration
        constitutional_config_path: Path to constitutional principles
        device: Target device
        
    Returns:
        Configured CAIPipeline instance
    """
    pipeline = CAIPipeline(
        constitutional_config_path=constitutional_config_path,
        device=device
    )
    
    # Load model (with fallback to mock)
    try:
        pipeline.load_model(model_key)
    except Exception as e:
        logger.warning(f"Failed to load model, using mock generation: {e}")
    
    return pipeline


if __name__ == "__main__":
    # Test the CAI pipeline
    print("Testing CAI Pipeline...")
    
    # Create pipeline
    pipeline = CAIPipeline()
    
    # Test with mock data (no model loading for quick test)
    test_prompts = [
        "What is the best way to handle disagreements?",
        "Should everyone follow the same moral principles?"
    ]
    
    test_responses = [
        "You should always argue until you win the disagreement.",
        "Yes, there is one correct moral system that everyone must follow."
    ]
    
    # Create preference pairs
    print("Creating preference pairs...")
    preference_pairs = pipeline.create_preference_pairs(test_prompts, test_responses)
    
    print(f"Created {len(preference_pairs)} preference pairs")
    
    # Show example
    if preference_pairs:
        example = preference_pairs[0]
        print(f"\nExample preference pair:")
        print(f"Prompt: {example.prompt}")
        print(f"Rejected: {example.rejected}")
        print(f"Chosen: {example.chosen}")
        print(f"Principle: {example.principle}")
    
    # Save to file
    output_path = Path(__file__).parent.parent.parent / "results" / "test_preference_pairs.jsonl"
    pipeline.save_preference_pairs(preference_pairs, str(output_path))
    
    # Get stats
    stats = pipeline.get_principle_stats(preference_pairs)
    print(f"\nPrinciple usage stats: {stats}")
    
    print("CAI Pipeline test completed!")
