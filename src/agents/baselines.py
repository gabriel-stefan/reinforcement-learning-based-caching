import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from src.environments.core.cache_storage import CacheItem

class EvictionPolicy(ABC):
    def __init__(self):
        self.name = "base"
        
    @abstractmethod
    def select_victim(self, items: List[CacheItem], current_timestamp: float) -> Optional[CacheItem]:
        pass

class RandomPolicy(EvictionPolicy):
    def __init__(self):
        self.name = "random"
        
    def select_victim(self, items: List[CacheItem], current_timestamp: float) -> Optional[CacheItem]:
        if not items:
            return None
        return random.choice(items)

class LRUPolicy(EvictionPolicy):
    def __init__(self):
        self.name = "lru"
        
    def select_victim(self, items: List[CacheItem], current_timestamp: float) -> Optional[CacheItem]:
        if not items:
            return None
        return min(items, key=lambda x: x.last_access)

class LFUPolicy(EvictionPolicy):
    def __init__(self):
        self.name = "lfu"
        
    def select_victim(self, items: List[CacheItem], current_timestamp: float) -> Optional[CacheItem]:
        if not items:
            return None
        return min(items, key=lambda x: (x.frequency, x.last_access))

class FIFOPolicy(EvictionPolicy):
    def __init__(self):
        self.name = "fifo"
        
    def select_victim(self, items: List[CacheItem], current_timestamp: float) -> Optional[CacheItem]:
        if not items:
            return None
        return min(items, key=lambda x: x.first_access)

class SizePolicy(EvictionPolicy):
    def __init__(self):
        self.name = "size"
        
    def select_victim(self, items: List[CacheItem], current_timestamp: float) -> Optional[CacheItem]:
        if not items:
            return None
        return max(items, key=lambda x: x.size)

POLICIES: Dict[str, type] = {
    'random': RandomPolicy,
    'lru': LRUPolicy,
    'lfu': LFUPolicy,
    'fifo': FIFOPolicy,
    'size': SizePolicy,
}

def get_policy(name: str) -> EvictionPolicy:
    name = name.lower()
    if name not in POLICIES:
        raise ValueError(f"Unknown policy '{name}'. Available: {list(POLICIES.keys())}")
    return POLICIES[name]()
