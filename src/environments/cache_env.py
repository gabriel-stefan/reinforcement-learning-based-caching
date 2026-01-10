import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, List
from collections import defaultdict

from .core.network_topology import (
    NetworkTopology, 
    HierarchicalLookupResult, 
    create_simple_hierarchy
)
from .core.cache_storage import CacheItem


class CacheEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        data_loader,
        topology: Optional[NetworkTopology] = None,
        render_mode: Optional[str] = None,
        include_static_features: bool = False,
    ):
        
        super().__init__()
        
        if topology is None:
            topology = create_simple_hierarchy()
        self.topology = topology
        self.nodes = topology.get_cache_node_ids()
        self.num_nodes = len(self.nodes)
        
        self.render_mode = render_mode
        self.loader = data_loader
        self.include_static = include_static_features
        
        self._max_latency = topology.calculate_origin_fetch_time(1024 * 1024)
        self._max_bandwidth = max(topology.get_node(n).bandwidth for n in self.nodes)
        self._num_tiers = topology.get_cache_tier_count()
        self._static_features = self._compute_static_features()
        
        self.request: Optional[Tuple] = None
        self._items: Dict[str, List[CacheItem]] = {}
        self._last_lookup: Optional[HierarchicalLookupResult] = None
        
        self._url_freq: Dict[str, int] = defaultdict(int)
        
        self.total_requests = 0
        self.total_hits = 0
        self.hit_bytes = 0
        self.total_bytes = 0
        self.tier_hits: Dict[int, int] = {}
        self.node_hits: Dict[str, int] = {}
        self.episode_reward = 0.0
        self.step_count = 0
        self._total_latency = 0.0
        self._origin_bytes = 0
        
        self.observation_space = self._make_obs_space()
        # new action space: 3 actions 
        # 0: cache at edge
        # 1: cache at regional 
        # 2: skip cache
        self.action_space = spaces.Discrete(3)
    
    def _compute_static_features(self) -> Dict[str, np.ndarray]:
        static = {}
        for node_id in self.nodes:
            node = self.topology.get_node(node_id)
            tier = self.topology._configs[node_id].tier
            static[node_id] = np.array([
                node.latency / self._max_latency,
                node.bandwidth / self._max_bandwidth,
                tier / max(1, self._num_tiers)
            ], dtype=np.float32)
        return static
    
    def _calc_obs_dim(self) -> int:
        per_node_dynamic = 6  
        per_node_static = 3 if self.include_static else 0
        per_node = per_node_dynamic + per_node_static
        request = 2 
        global_feat = 3 
        return self.num_nodes * per_node + request + global_feat
    
    def _make_obs_space(self) -> spaces.Box:
        dim = self._calc_obs_dim()
        return spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.loader.reset()
        self.topology.clear_all()
        self._items = {n: [] for n in self.nodes}
        self._last_lookup = None
        
        self._url_freq.clear()
        
        self.total_requests = 0
        self.total_hits = 0
        self.hit_bytes = 0
        self.total_bytes = 0
        self.tier_hits = {i: 0 for i in range(self._num_tiers + 1)}
        self.node_hits = {n: 0 for n in self.nodes}
        self.episode_reward = 0.0
        self.step_count = 0
        self._total_latency = 0.0
        self._origin_bytes = 0
        
        self._next_request()
        
        return self._get_obs(), {'step': 0}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        
        if self.request is None:
            return self._get_obs(), 0.0, True, False, {'done': True}
        
        reward, done, info = self._apply(action)
        self.episode_reward += reward
        
        if info.get('processed'):
            self._update_metrics(info)
            self._next_request()
            if self.request is None:
                done = True
        
        info['step'] = self.step_count
        info['episode_reward'] = self.episode_reward
        
        return self._get_obs(), reward, done, False, info
    
    def _get_obs(self) -> np.ndarray:
        dim = self._calc_obs_dim()
        if self.request is None:
            return np.zeros(dim, dtype=np.float32)
        
        url, size, timestamp, *_ = self.request
        obs = []
        
        for node_id in self.nodes:
            node = self.topology.get_node(node_id)
            
            obs.append(node.get_occupancy())
            
            items = node.cache.get_all_items()
            self._items[node_id] = items
            
            if items:
                if len(items) > 500:
                    indices = np.random.choice(len(items), 500, replace=False)
                    sampled_items = [items[i] for i in indices]
                else:
                    sampled_items = items
                    
                num_items = min(1.0, len(items) / 100)
                sizes = [i.size for i in sampled_items]
                avg_size = self._norm_size(np.mean(sizes))
                avg_freq = self._norm_freq(int(np.mean([i.frequency for i in sampled_items])))
                avg_recency = np.mean([
                    np.exp(-0.001 * max(0, timestamp - i.last_access)) 
                    for i in sampled_items
                ])
            else:
                num_items = avg_size = avg_freq = avg_recency = 0.0
            
            obs.extend([num_items, avg_size, avg_freq, avg_recency])
            
            free_space = node.get_free_space()
            cache_pressure = min(2.0, size / max(1, free_space))
            obs.append(cache_pressure / 2.0)  
            
            if self.include_static:
                obs.extend(self._static_features[node_id].tolist())
        
        obs.append(self._norm_size(size))
        obs.append(self._get_freq_hint(url))
        
        obs.append(self.total_hits / max(1, self.total_requests))
        
        total_tier_hits = sum(self.tier_hits.values()) or 1
        edge_ratio = self.tier_hits.get(0, 0) / total_tier_hits
        regional_ratio = self.tier_hits.get(1, 0) / total_tier_hits
        obs.extend([edge_ratio, regional_ratio])
            
        return np.array(obs, dtype=np.float32)
    
    def _get_freq_hint(self, url: str) -> float:
        freq = self._url_freq.get(url, 0)
        return float(np.log1p(freq) / np.log1p(100))
    
    def _update_freq_history(self, url: str):
        self._url_freq[url] += 1
        
    def _apply(self, action: int) -> Tuple[float, bool, Dict]:
        url, size, timestamp, *_ = self.request
        info = {'processed': False, 'hit': False, 'size': size}
        
        lookup = self.topology.lookup(url, size, timestamp)
        self._last_lookup = lookup
        info['latency'] = lookup.total_latency_ms
        self._total_latency += lookup.total_latency_ms
        
        if lookup.from_origin:
            self._origin_bytes += size
        
        if lookup.hit and not lookup.from_origin:
            reward = self._reward(lookup, size, 0, [])
            info.update({
                'processed': True,
                'hit': True,
                'node': lookup.hit_node_id,
                'tier': lookup.hit_tier,
                'savings': self.topology.calculate_origin_fetch_time(size) - lookup.total_latency_ms
            })
            return reward, False, info
        
        if action == 2:  
            info['processed'] = True
            info['skipped'] = True
            return -1.0, False, info  
        
        target_node = None
        target_node_id = None
        
        if action == 0:  # Cache at edge (tier 0)
            target_node = self.topology.get_node(self.nodes[0])
            target_node_id = self.nodes[0]
        elif action == 1:  # Cache at regional (tier 1)
            target_node = self.topology.get_node(self.nodes[1])
            target_node_id = self.nodes[1]
        
        # evict with lfu 
        evicted_items_list = []
        
        # max 20 evictions to prevent infinite loops
        for _ in range(20):
            if target_node.get_free_space() >= size:
                break
                
            items = target_node.cache.get_all_items()
            if not items:
                break 
            evicted_item = min(items, key=lambda x: x.frequency) # lfu strategy proven best on this trace
            
            target_node.evict(evicted_item.url)
            evicted_items_list.append(evicted_item)
        
        if target_node.get_free_space() >= size:
            target_node.add(url, size, timestamp)
            
            # latency-based rewards
            reward = self._reward(lookup, size, len(evicted_items_list), evicted_items_list)
            
            # small bonus for successful caching to encourage exploration
            reward += 0.1
            
            info['processed'] = True
            info['cached_at'] = target_node_id
            info['evicted_items'] = len(evicted_items_list)
            info['action_type'] = ['EDGE', 'REGIONAL', 'SKIP'][action]
            return reward, False, info
        else:
            info['processed'] = True
            info['failed_to_cache'] = True
            return -1.0, False, info #  failure to cache is a miss
    
    def _reward(self, lookup: HierarchicalLookupResult, size: int, 
                evictions: int, evicted_items: Optional[List[CacheItem]] = None
                ) -> float:

        # reward formula = (origin_latency - actual_latency) / origin_latency * scale
        
        origin_latency = self.topology.calculate_origin_fetch_time(size)
        
        hit = lookup.hit and not lookup.from_origin
        if hit:
            actual_latency = lookup.total_latency_ms
        else:
            actual_latency = origin_latency
    
        savings_ratio = (origin_latency - actual_latency) / max(origin_latency, 1e-6)
        
        reward = savings_ratio * 10.0
        
 
        if evicted_items:
            for item in evicted_items:
                freq_penalty = min(item.frequency / 10.0, 1.0) * 0.1
                reward -= freq_penalty
        
        if not hit:
            reward = 0.0 
        
        return reward
    

    
    
    def _update_metrics(self, info: Dict):
        self.total_requests += 1
        self.total_bytes += info.get('size', 0)
        
        if self.request:
            url = self.request[0]
            self._update_freq_history(url)
        
        if info.get('hit'):
            self.total_hits += 1
            self.hit_bytes += info.get('size', 0)
            
            tier = info.get('tier')
            if tier is not None:
                self.tier_hits[tier] = self.tier_hits.get(tier, 0) + 1
            
            node = info.get('node')
            if node:
                self.node_hits[node] = self.node_hits.get(node, 0) + 1
    
    def _next_request(self):
        self.request = self.loader.get_next()
    
    def get_metrics(self) -> Dict:
        metrics = {
            'hit_rate': self.total_hits / max(1, self.total_requests),
            'byte_hit_rate': self.hit_bytes / max(1, self.total_bytes),
            'requests': self.total_requests,
            'hits': self.total_hits,
            'reward': self.episode_reward,
            'steps': self.step_count,
            'avg_latency': self._total_latency / max(1, self.total_requests),
            'origin_bandwidth': self._origin_bytes / max(1, self.step_count)  # bytes per step
        }
        
        total_tier_hits = sum(self.tier_hits.values()) or 1
        for tier, hits in self.tier_hits.items():
            metrics[f'tier_{tier}_hits'] = hits
            metrics[f'tier_{tier}_rate'] = hits / total_tier_hits
        
        for node_id in self.nodes:
            node = self.topology.get_node(node_id)
            metrics[f'{node_id}_occupancy'] = node.get_occupancy()
            metrics[f'{node_id}_hits'] = self.node_hits.get(node_id, 0)
            metrics[f'{node_id}_items'] = len(self._items.get(node_id, []))
        
        return metrics
        
    def _norm_size(self, size: int, max_size: int = 5_000_000) -> float:
        return float(np.log1p(size) / np.log1p(max_size))
    
    def _norm_freq(self, freq: int, max_freq: int = 100) -> float:
        return float(np.log1p(freq) / np.log1p(max_freq))
        
    def render(self):
        if self.render_mode == 'human':
            print(self._render_ansi())
        return self._render_ansi() if self.render_mode == 'ansi' else None
    
    def _render_ansi(self) -> str:
        m = self.get_metrics()
        lines = [
            f"Step: {self.step_count}",
            f"Hit Rate: {m['hit_rate']:.2%}",
            f"Byte Hit Rate: {m['byte_hit_rate']:.2%}",
            f"Requests: {self.total_requests}",
            f"Episode Reward: {self.episode_reward:.2f}",
            f"",
            f"Nodes:"
        ]
        
        for node_id in self.nodes:
            node = self.topology.get_node(node_id)
            tier = self.topology._configs[node_id].tier
            lines.append(f"  {node_id} (tier {tier}): {node.get_occupancy():.1%} full, "
                        f"{len(self._items.get(node_id, []))} items, "
                        f"{self.node_hits.get(node_id, 0)} hits")
        
        if self.request:
            url, size, *_ = self.request
            lines.append(f"")
            lines.append(f"Current: {url[:40]}... ({size} bytes)")
        
        return "\n".join(lines)
    
    def close(self):
        pass



def make_cache_env(data_loader, **kwargs) -> CacheEnv:
    return CacheEnv(data_loader=data_loader, **kwargs)


def make_hierarchy_env(
    data_loader,
    edge_mb: int = 10,
    regional_mb: int = 100,
    origin_latency: float = 100.0,
    **kwargs
) -> CacheEnv:
    topology = create_simple_hierarchy(
        edge_capacity_mb=edge_mb,
        regional_capacity_mb=regional_mb,
        origin_latency_ms=origin_latency
    )
    return CacheEnv(data_loader=data_loader, topology=topology, **kwargs)
