from dataclasses import dataclass
from typing import Dict, List, Optional
from .network_node import NetworkNode
from .cache_storage import CacheItem


@dataclass
class NodeConfig:
    node_id: str
    capacity_mb: int          
    latency_ms: float         
    bandwidth_mbps: float     
    tier: int = 0            
    is_origin: bool = False
    
    def __post_init__(self):
        self.capacity_bytes = (1024 ** 4) if self.capacity_mb == -1 else (self.capacity_mb * 1024 * 1024)


@dataclass 
class HierarchicalLookupResult:
    hit: bool
    hit_node_id: Optional[str]   
    hit_tier: Optional[int]     
    total_latency_ms: float      
    item_size: int = 0
    from_origin: bool = False    
    
    @property
    def is_cache_hit(self) -> bool:
        return self.hit and not self.from_origin


class NetworkTopology:
    def __init__(self):
        self._nodes: Dict[str, NetworkNode] = {}
        self._configs: Dict[str, NodeConfig] = {}
        self.tiers: Dict[int, List[str]] = {}  # tier -> [node_ids]
        self.origin_node_id: Optional[str] = None
    
    def add_node(self, config: NodeConfig) -> NetworkNode:
        node = NetworkNode(
            node_id=config.node_id,
            cache_capacity=config.capacity_bytes,
            bandwidth_mbps=config.bandwidth_mbps,
            latency_ms=config.latency_ms
        )
        
        self._nodes[config.node_id] = node
        self._configs[config.node_id] = config
        
        if config.is_origin:
            self.origin_node_id = config.node_id
        
        if config.tier not in self.tiers:
            self.tiers[config.tier] = []
        self.tiers[config.tier].append(config.node_id)
        
        return node
    
    def get_node(self, node_id: str) -> Optional[NetworkNode]:
        return self._nodes.get(node_id)
    
    def get_cache_node_ids(self) -> List[str]:
        return [nid for nid, cfg in self._configs.items() if not cfg.is_origin]
    
    def get_cache_tier_count(self) -> int:
        if not self.origin_node_id:
            return len(self.tiers)
        origin_tier = self._configs[self.origin_node_id].tier
        return len([t for t in self.tiers if t != origin_tier])
    
    def lookup(self, url: str, size: int, timestamp: float) -> HierarchicalLookupResult:
        latency = 0.0
        
        for tier in sorted(self.tiers.keys()):
            for node_id in self.tiers[tier]:
                node = self._nodes[node_id]
                config = self._configs[node_id]
                latency += node.latency
                
                if config.is_origin:
                    latency += size / node._bytes_per_ms
                    return HierarchicalLookupResult(
                        hit=True, hit_node_id=node_id, hit_tier=tier,
                        total_latency_ms=latency, item_size=size, from_origin=True
                    )
                
                if node.contains(url):
                    node.request(url, size, timestamp)  
                    latency += size / node._bytes_per_ms
                    return HierarchicalLookupResult(
                        hit=True, hit_node_id=node_id, hit_tier=tier,
                        total_latency_ms=latency, item_size=size, from_origin=False
                    )
        
        return HierarchicalLookupResult(
            hit=False, hit_node_id=None, hit_tier=None,
            total_latency_ms=latency, item_size=size, from_origin=False
        )
    
    def clear_all(self):
        for node in self._nodes.values():
            node.clear()
    
    def calculate_origin_fetch_time(self, size: int) -> float:
        if not self.origin_node_id:
            return 100.0
        
        origin = self._nodes[self.origin_node_id]
        origin_tier = self._configs[self.origin_node_id].tier
        
        latency = sum(
            self._nodes[self.tiers[t][0]].latency 
            for t in sorted(self.tiers.keys()) if t < origin_tier and self.tiers[t]
        )
        latency += origin.latency + size / origin._bytes_per_ms
        return latency
    
    def __repr__(self) -> str:
        parts = []
        for tier in sorted(self.tiers.keys()):
            nodes = self.tiers[tier]
            suffix = " [ORIGIN]" if any(self._configs[n].is_origin for n in nodes) else ""
            parts.append(f"Tier {tier}: {nodes}{suffix}")
        return f"NetworkTopology({', '.join(parts)})"


def create_simple_hierarchy(
    edge_capacity_mb: int = 10,
    edge_latency_ms: float = 1.0,
    edge_bandwidth_mbps: float = 1000.0,
    regional_capacity_mb: int = 100,
    regional_latency_ms: float = 10.0,
    regional_bandwidth_mbps: float = 500.0,
    origin_latency_ms: float = 50.0,
    origin_bandwidth_mbps: float = 100.0
) -> NetworkTopology:
    
    topology = NetworkTopology()
    
    topology.add_node(NodeConfig(
        node_id='edge',
        capacity_mb=edge_capacity_mb,
        latency_ms=edge_latency_ms,
        bandwidth_mbps=edge_bandwidth_mbps,
        tier=0
    ))
    
    topology.add_node(NodeConfig(
        node_id='regional',
        capacity_mb=regional_capacity_mb,
        latency_ms=regional_latency_ms,
        bandwidth_mbps=regional_bandwidth_mbps,
        tier=1
    ))
    
    topology.add_node(NodeConfig(
        node_id='origin',
        capacity_mb=-1,
        latency_ms=origin_latency_ms,
        bandwidth_mbps=origin_bandwidth_mbps,
        tier=2,
        is_origin=True
    ))
    
    return topology
