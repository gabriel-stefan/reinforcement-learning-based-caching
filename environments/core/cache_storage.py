from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class CacheItem:
    url: str
    size: int  
    first_access: float 
    last_access: float  
    frequency: int = 1 
    
    def update_access(self, timestamp: float):
        self.last_access = timestamp
        self.frequency += 1


class CacheStorage:
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.current_size = 0
        self.items: Dict[str, CacheItem] = {}
        
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._hit_bytes = 0
        self._total_bytes = 0
    
    
    def contains(self, url: str) -> bool:
        return url in self.items
    
    def add(self, url: str, size: int, timestamp: float) -> bool:
        if size > self.capacity:
            return False
            
        if self.get_free_space() < size:
            return False
        
        if url in self.items:
            self.access(url, timestamp)
            return True
        
        item = CacheItem(
            url=url,
            size=size,
            first_access=timestamp,
            last_access=timestamp,
            frequency=1
        )
        self.items[url] = item
        self.current_size += size
        return True
    
    def evict(self, url: str) -> int:
        if url not in self.items:
            return 0
        
        item = self.items.pop(url)
        self.current_size -= item.size
        self._evictions += 1
        return item.size
    
    def access(self, url: str, timestamp: float) -> bool:
        if url in self.items:
            self.items[url].update_access(timestamp)
            self._hits += 1
            self._hit_bytes += self.items[url].size
            return True
        else:
            self._misses += 1
            return False
    
    def record_request(self, size: int):
        self._total_bytes += size
    
    def clear(self):
        self.items.clear()
        self.current_size = 0
        self._reset_stats()
    
    def _reset_stats(self):
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._hit_bytes = 0
        self._total_bytes = 0
        
    def get_occupancy(self) -> float:
        if self.capacity == 0:
            return 1.0
        return self.current_size / self.capacity
    
    def get_free_space(self) -> int:
        return self.capacity - self.current_size
    
    def get_item(self, url: str) -> Optional[CacheItem]:
        return self.items.get(url)
    
    def get_all_items(self) -> List[CacheItem]:
        return list(self.items.values())
    
    def get_candidates(self, n: int, current_time: float) -> List[CacheItem]:
        items = list(self.items.values())
        if len(items) <= n:
            return items
        
        n_lru = max(1, int(n * 0.34))
        n_lfu = max(1, int(n * 0.33))
        n_random = n - n_lru - n_lfu
        
        lru_sorted = sorted(items, key=lambda x: x.last_access)
        lru_candidates = lru_sorted[:n_lru]
        
        remaining = [i for i in items if i not in lru_candidates]
        lfu_sorted = sorted(remaining, key=lambda x: x.frequency)
        lfu_candidates = lfu_sorted[:n_lfu]
        
        remaining = [i for i in items if i not in lru_candidates and i not in lfu_candidates]
        random_candidates = random.sample(remaining, min(n_random, len(remaining)))
        candidates = lru_candidates + lfu_candidates + random_candidates
        
        while len(candidates) < n and len(lru_sorted) > len(candidates):
            for item in lru_sorted:
                if item not in candidates:
                    candidates.append(item)
                    if len(candidates) >= n:
                        break
        
        return candidates[:n]
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __repr__(self) -> str:
        return (f"CacheStorage(items={len(self.items)}, "
                f"size={self.current_size}/{self.capacity} bytes, "
                f"occupancy={self.get_occupancy():.1%})")
