"""AILuminate dataset loader for Constitutional AI."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


# Hazard categories available in AILuminate
PHYSICAL_HAZARDS = ['vcr', 'src', 'cse', 'ssh', 'iwp']
NON_PHYSICAL_HAZARDS = ['ipv', 'dfm', 'ncr', 'hte', 'prv']
CONTEXTUAL_HAZARDS = ['spc_ele', 'spc_fin', 'spc_hlt', 'sxc_prn']
ALL_HAZARDS = PHYSICAL_HAZARDS + NON_PHYSICAL_HAZARDS + CONTEXTUAL_HAZARDS

PERSONA_TYPES = ['normal', 'unskilled', 'skilled']


class AILuminateLoader:
    """
    Load and filter AILuminate benchmark prompts.
    
    The AILuminate dataset contains adversarial prompts designed to elicit
    unsafe responses across 14 hazard categories. Perfect for Constitutional AI
    where we need prompts that will violate constitutional principles.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the loader.
        
        Args:
            dataset_path: Path to AILuminate CSV file
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"AILuminate dataset not found: {self.dataset_path}")
        
        logger.info(f"Loading AILuminate dataset from: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        logger.info(f"Loaded {len(self.df)} prompts")
        
        # Standardize column names (handle different CSV formats)
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
    
    def get_prompts(
        self, 
        hazard_categories: Optional[List[str]] = None,
        persona_types: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
        random_state: int = 42
    ) -> List[Dict[str, str]]:
        """
        Get prompts filtered by hazard category and persona.
        
        Args:
            hazard_categories: List of hazard codes (e.g., ['vcr', 'cse', 'hte'])
                              None = all categories
            persona_types: List of persona types (e.g., ['normal', 'skilled'])
                          None = all personas
            n_samples: Number of samples to return (None = all)
            random_state: Random seed for sampling
        
        Returns:
            List of dicts with 'prompt', 'hazard_category', 'persona_type', etc.
        """
        df = self.df.copy()
        
        # Filter by hazard category
        if hazard_categories:
            # Validate hazard categories
            invalid = set(hazard_categories) - set(ALL_HAZARDS)
            if invalid:
                logger.warning(f"Invalid hazard categories: {invalid}")
            
            df = df[df['hazard'].isin(hazard_categories)]
            logger.info(f"Filtered to {len(df)} prompts in hazard categories: {hazard_categories}")
        
        # Filter by persona type
        if persona_types:
            # Validate persona types
            invalid = set(persona_types) - set(PERSONA_TYPES)
            if invalid:
                logger.warning(f"Invalid persona types: {invalid}")
            
            df = df[df['persona'].isin(persona_types)]
            logger.info(f"Filtered to {len(df)} prompts with persona types: {persona_types}")
        
        # Sample if requested
        if n_samples and len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=random_state)
            logger.info(f"Sampled {n_samples} prompts")
        
        # Convert to list of dicts
        prompts = []
        for _, row in df.iterrows():
            prompts.append({
                'prompt': row['prompt_text'],
                'hazard_category': row['hazard'],
                'persona_type': row['persona'],
                'locale': row.get('locale', 'en_US'),
                'prompt_id': row.get('release_prompt_id', ''),
                'prompt_hash': row.get('prompt_hash', ''),
            })
        
        return prompts
    
    def get_hazard_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of prompts across hazard categories.
        
        Returns:
            Dictionary mapping hazard category to count
        """
        return dict(self.df['hazard'].value_counts())
    
    def get_persona_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of prompts across persona types.
        
        Returns:
            Dictionary mapping persona type to count
        """
        return dict(self.df['persona'].value_counts())
    
    def get_available_hazards(self) -> Set[str]:
        """Get set of hazard categories present in the dataset."""
        return set(self.df['hazard'].unique())
    
    def get_available_personas(self) -> Set[str]:
        """Get set of persona types present in the dataset."""
        return set(self.df['persona'].unique())
    
    def print_statistics(self) -> None:
        """Print dataset statistics."""
        print(f"\n=== AILuminate Dataset Statistics ===")
        print(f"Total prompts: {len(self.df)}")
        print(f"\nHazard categories:")
        for hazard, count in sorted(self.get_hazard_distribution().items()):
            print(f"  {hazard}: {count}")
        print(f"\nPersona types:")
        for persona, count in sorted(self.get_persona_distribution().items()):
            print(f"  {persona}: {count}")


def load_ailuminate_prompts(
    dataset_path: str = "data/benchmarks/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv",
    hazard_categories: Optional[List[str]] = None,
    persona_types: Optional[List[str]] = None,
    n_samples: Optional[int] = None,
    random_state: int = 42
) -> List[str]:
    """
    Convenience function to load AILuminate prompts as a simple list.
    
    Args:
        dataset_path: Path to AILuminate CSV
        hazard_categories: Filter by hazard categories
        persona_types: Filter by persona types  
        n_samples: Number of samples to return
        random_state: Random seed
        
    Returns:
        List of prompt strings
    """
    loader = AILuminateLoader(dataset_path)
    prompt_dicts = loader.get_prompts(
        hazard_categories=hazard_categories,
        persona_types=persona_types,
        n_samples=n_samples,
        random_state=random_state
    )
    return [p['prompt'] for p in prompt_dicts]


if __name__ == "__main__":
    # Test the loader
    import sys
    
    # Try to load from default location
    default_path = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "ailuminate" / "airr_official_1.0_demo_en_us_prompt_set_release.csv"
    
    if not default_path.exists():
        print(f"AILuminate dataset not found at: {default_path}")
        print("Please run: git submodule update --init --recursive")
        sys.exit(1)
    
    # Load and print statistics
    loader = AILuminateLoader(str(default_path))
    loader.print_statistics()
    
    # Test filtering
    print("\n=== Test: Physical hazards only ===")
    physical_prompts = loader.get_prompts(
        hazard_categories=PHYSICAL_HAZARDS,
        n_samples=5
    )
    for i, p in enumerate(physical_prompts, 1):
        print(f"\n{i}. [{p['hazard_category']}] {p['prompt'][:100]}...")
    
    print("\n=== Test: Skilled adversarial persona only ===")
    skilled_prompts = loader.get_prompts(
        persona_types=['skilled'],
        n_samples=5
    )
    for i, p in enumerate(skilled_prompts, 1):
        print(f"\n{i}. [{p['hazard_category']}] {p['prompt'][:100]}...")

