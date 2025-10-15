"""Train/test split manager for Constitutional AI datasets."""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class SplitManager:
    """
    Manage train/test splits for Constitutional AI datasets.
    
    The split configuration is stored in a JSON file so that the same
    split can be used consistently across data generation, training,
    and evaluation scripts.
    """
    
    def __init__(self, config_path: str = "data/splits/default_split.json"):
        """
        Initialize split manager.
        
        Args:
            config_path: Path to split configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config: Dict = {}
        if self.config_path.exists():
            self.load()
    
    def create_split(
        self,
        prompt_ids: List[str],
        test_size: float = 0.1,
        random_state: int = 42,
        stratify_by: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[Set[str], Set[str]]:
        """
        Create a train/test split and save configuration.
        
        Args:
            prompt_ids: List of unique prompt identifiers
            test_size: Fraction of data for test set (0.0-1.0)
            random_state: Random seed for reproducibility
            stratify_by: Optional list of stratification keys (not implemented yet)
            metadata: Optional metadata to store with split
            
        Returns:
            Tuple of (train_ids, test_ids) as sets
        """
        import random
        
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        # Shuffle with fixed seed
        rng = random.Random(random_state)
        shuffled_ids = prompt_ids.copy()
        rng.shuffle(shuffled_ids)
        
        # Split
        n_test = max(1, int(len(shuffled_ids) * test_size))
        test_ids = set(shuffled_ids[:n_test])
        train_ids = set(shuffled_ids[n_test:])
        
        # Store configuration
        self.config = {
            'version': '1.0',
            'created_at': self._timestamp(),
            'random_state': random_state,
            'test_size': test_size,
            'n_total': len(prompt_ids),
            'n_train': len(train_ids),
            'n_test': len(test_ids),
            'train_ids': sorted(train_ids),
            'test_ids': sorted(test_ids),
            'metadata': metadata or {},
            'split_hash': self._compute_hash(train_ids, test_ids)
        }
        
        self.save()
        
        logger.info(f"Created split: {len(train_ids)} train, {len(test_ids)} test")
        logger.info(f"Split configuration saved to: {self.config_path}")
        
        return train_ids, test_ids
    
    def get_split(self) -> Tuple[Set[str], Set[str]]:
        """
        Get the train/test split from saved configuration.
        
        Returns:
            Tuple of (train_ids, test_ids) as sets
        """
        if not self.config:
            raise ValueError(f"No split configuration found at {self.config_path}")
        
        train_ids = set(self.config['train_ids'])
        test_ids = set(self.config['test_ids'])
        
        return train_ids, test_ids
    
    def is_train(self, prompt_id: str) -> bool:
        """Check if a prompt ID is in the training set."""
        train_ids, _ = self.get_split()
        return prompt_id in train_ids
    
    def is_test(self, prompt_id: str) -> bool:
        """Check if a prompt ID is in the test set."""
        _, test_ids = self.get_split()
        return prompt_id in test_ids
    
    def filter_train(self, items: List[Dict]) -> List[Dict]:
        """
        Filter a list of items to only include training set.
        
        Args:
            items: List of dicts, each with 'prompt_id' or 'id' key
            
        Returns:
            Filtered list containing only training items
        """
        train_ids, _ = self.get_split()
        return [item for item in items if self._get_id(item) in train_ids]
    
    def filter_test(self, items: List[Dict]) -> List[Dict]:
        """
        Filter a list of items to only include test set.
        
        Args:
            items: List of dicts, each with 'prompt_id' or 'id' key
            
        Returns:
            Filtered list containing only test items
        """
        _, test_ids = self.get_split()
        return [item for item in items if self._get_id(item) in test_ids]
    
    def save(self) -> None:
        """Save split configuration to disk."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Split configuration saved: {self.config_path}")
    
    def load(self) -> None:
        """Load split configuration from disk."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        logger.info(f"Loaded split configuration: {self.config_path}")
        logger.info(f"  Train: {self.config['n_train']} samples")
        logger.info(f"  Test: {self.config['n_test']} samples")
    
    def print_statistics(self) -> None:
        """Print split statistics."""
        if not self.config:
            print("No split configuration loaded")
            return
        
        print(f"\n=== Split Statistics ===")
        print(f"Configuration: {self.config_path}")
        print(f"Created: {self.config['created_at']}")
        print(f"Random state: {self.config['random_state']}")
        print(f"Test size: {self.config['test_size']:.1%}")
        print(f"Total samples: {self.config['n_total']}")
        print(f"Train samples: {self.config['n_train']} ({self.config['n_train']/self.config['n_total']:.1%})")
        print(f"Test samples: {self.config['n_test']} ({self.config['n_test']/self.config['n_total']:.1%})")
        
        if self.config.get('metadata'):
            print(f"\nMetadata:")
            for key, value in self.config['metadata'].items():
                print(f"  {key}: {value}")
    
    def _get_id(self, item: Dict) -> str:
        """Extract ID from item dict."""
        return item.get('prompt_id') or item.get('id') or item.get('prompt_hash', '')
    
    def _timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _compute_hash(self, train_ids: Set[str], test_ids: Set[str]) -> str:
        """Compute hash of split for validation."""
        combined = '|'.join(sorted(train_ids)) + '||' + '|'.join(sorted(test_ids))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


def create_train_test_split(
    prompt_ids: List[str],
    test_size: float = 0.1,
    random_state: int = 42,
    config_path: str = "data/splits/default_split.json",
    metadata: Optional[Dict] = None
) -> Tuple[Set[str], Set[str]]:
    """
    Convenience function to create and save a train/test split.
    
    Args:
        prompt_ids: List of unique prompt identifiers
        test_size: Fraction for test set
        random_state: Random seed
        config_path: Where to save split configuration
        metadata: Optional metadata
        
    Returns:
        Tuple of (train_ids, test_ids) as sets
    """
    manager = SplitManager(config_path)
    return manager.create_split(
        prompt_ids=prompt_ids,
        test_size=test_size,
        random_state=random_state,
        metadata=metadata
    )


if __name__ == "__main__":
    # Test the split manager
    import sys
    
    # Create a test split
    print("=== Creating test split ===")
    test_ids = [f"prompt_{i}" for i in range(100)]
    
    manager = SplitManager("data/splits/test_split.json")
    train_ids, test_ids = manager.create_split(
        prompt_ids=test_ids,
        test_size=0.2,
        random_state=42,
        metadata={'dataset': 'test', 'version': '1.0'}
    )
    
    print(f"Train: {len(train_ids)}")
    print(f"Test: {len(test_ids)}")
    
    # Test loading
    print("\n=== Loading split ===")
    manager2 = SplitManager("data/splits/test_split.json")
    train_ids2, test_ids2 = manager2.get_split()
    
    assert train_ids == train_ids2
    assert test_ids == test_ids2
    print("✓ Split loaded correctly")
    
    # Print statistics
    manager2.print_statistics()
    
    # Test filtering
    print("\n=== Testing filtering ===")
    items = [{'prompt_id': f'prompt_{i}', 'data': f'test_{i}'} for i in range(100)]
    train_items = manager2.filter_train(items)
    test_items = manager2.filter_test(items)
    
    print(f"Train items: {len(train_items)}")
    print(f"Test items: {len(test_items)}")
    assert len(train_items) + len(test_items) == len(items)
    print("✓ Filtering works correctly")

