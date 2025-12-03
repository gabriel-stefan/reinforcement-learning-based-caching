from dataclasses import dataclass
from typing import Optional, List
from .cache_storage import CacheStorage, CacheItem

@dataclass
class RequestResult:
    hit: bool
    time_cost: float 
    source: Optional[str] 
    size: int = 0
    
    def __repr__(self):
        status = "HIT" if self.hit else "MISS"
        return f"RequestResult({status}, time={self.time_cost:.2f}ms, source={self.source})"


class NetworkNode:
    
    def __init__(self,
                 node_id: str,
                 cache_capacity: int,
                 bandwidth_mbps: float = 1000.0,  
                 latency_ms: float = 1.0):        
    
        self.node_id = node_id
        self.cache = CacheStorage(cache_capacity)
        self.bandwidth = bandwidth_mbps  # Mbps
        self.latency = latency_ms  # ms
        
        self._bytes_per_ms = bandwidth_mbps * 125
    
    def calculate_transfer_time(self, size_bytes: int) -> float:
        transfer_time = size_bytes / self._bytes_per_ms
        return self.latency + transfer_time
    
    def request(self, url: str, size: int, timestamp: float) -> RequestResult:
        self.cache.record_request(size)
        
        if self.cache.contains(url): #cache hit
            self.cache.access(url, timestamp)
            item = self.cache.get_item(url)
            time_cost = self.calculate_transfer_time(item.size)
            return RequestResult(
                hit=True,
                time_cost=time_cost,
                source=self.node_id,
                size=item.size
            )
        else: #cache miss - fetch from origin
            return RequestResult(
                hit=False,
                time_cost=0,
                source=None,
                size=size
            )
        
    def add(self, url: str, size: int, timestamp: float) -> bool:
        return self.cache.add(url, size, timestamp)
    
    def evict(self, url: str) -> int:
        return self.cache.evict(url)
    
    def contains(self, url: str) -> bool:
        return self.cache.contains(url)
    
    def get_free_space(self) -> int:
        return self.cache.get_free_space()
    
    def get_occupancy(self) -> float:
        return self.cache.get_occupancy()
    
    def clear(self):
        self.cache.clear()
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def __repr__(self) -> str:
        return (f"NetworkNode(id={self.node_id}, "
                f"cache={self.cache}, "
                f"bandwidth={self.bandwidth}Mbps, "
                f"latency={self.latency}ms)")
