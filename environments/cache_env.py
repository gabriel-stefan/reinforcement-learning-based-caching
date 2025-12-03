import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
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
        alpha: float = 1.0,    
        beta: float = 0.5,     
        gamma: float = 0.1,
        k_candidates: int = 20,
        render_mode: Optional[str] = None,
        include_static_features: bool = False,
        freq_history_window: int = 10000,
        adaptive_rewards: bool = False,
    ):
        
        super().__init__()
        
        if topology is None:
            topology = create_simple_hierarchy()
        self.topology = topology
        self.nodes = topology.get_cache_node_ids()
        self.num_nodes = len(self.nodes)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.adaptive_rewards = adaptive_rewards
        
        self.k_candidates = k_candidates
        self.render_mode = render_mode
        self.loader = data_loader
        
        self.include_static = include_static_features
        self.freq_history_window = freq_history_window
        
        self._max_latency = topology.calculate_origin_fetch_time(1024 * 1024)
        self._max_bandwidth = max(topology.get_node(n).bandwidth for n in self.nodes)
        self._num_tiers = topology.get_cache_tier_count()
        self._static_features = self._compute_static_features()
        
        self.request: Optional[Tuple] = None
        self._candidates: Dict[str, List[CacheItem]] = {}  # Current eviction candidates per node
        self._items: Dict[str, List[CacheItem]] = {}
        self._last_lookup: Optional[HierarchicalLookupResult] = None
        
        self._url_freq: Dict[str, int] = defaultdict(int)
        self._url_history: List[str] = []
        
        self.total_requests = 0
        self.total_hits = 0
        self.hit_bytes = 0
        self.total_bytes = 0
        self.tier_hits: Dict[int, int] = {}
        self.node_hits: Dict[str, int] = {}
        self.episode_reward = 0.0
        self.step_count = 0
        
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Discrete(self.num_nodes * self.k_candidates + 1)
    
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
        candidate_features = self.num_nodes * self.k_candidates * 5
        return self.num_nodes * per_node + request + global_feat + candidate_features
    
    def _make_obs_space(self) -> spaces.Box:
        dim = self._calc_obs_dim()
        return spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.loader.reset()
        self.topology.clear_all()
        self._items = {n: [] for n in self.nodes}
        self._candidates = {n: [] for n in self.nodes}
        self._last_lookup = None
        
        self._url_freq.clear()
        self._url_history.clear()
        
        self.total_requests = 0
        self.total_hits = 0
        self.hit_bytes = 0
        self.total_bytes = 0
        self.tier_hits = {i: 0 for i in range(self._num_tiers + 1)}
        self.node_hits = {n: 0 for n in self.nodes}
        self.episode_reward = 0.0
        self.step_count = 0
        
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
        
        node_size_percentiles = {}  
        for node_id in self.nodes:
            node = self.topology.get_node(node_id)
            
            obs.append(node.get_occupancy())
            
            items = node.cache.get_all_items()
            self._items[node_id] = items
            
            if items:
                num_items = min(1.0, len(items) / 100)
                sizes = [i.size for i in items]
                avg_size = self._norm_size(np.mean(sizes))
                avg_freq = self._norm_freq(int(np.mean([i.frequency for i in items])))
                avg_recency = np.mean([
                    np.exp(-0.001 * max(0, timestamp - i.last_access)) 
                    for i in items
                ])
                sorted_sizes = np.sort(sizes)
                node_size_percentiles[node_id] = {
                    'p10': sorted_sizes[int(len(sorted_sizes) * 0.1)] if len(sorted_sizes) > 0 else 0,
                    'p50': sorted_sizes[int(len(sorted_sizes) * 0.5)] if len(sorted_sizes) > 0 else 0,
                    'p90': sorted_sizes[int(len(sorted_sizes) * 0.9)] if len(sorted_sizes) > 0 else 0,
                }
            else:
                num_items = avg_size = avg_freq = avg_recency = 0.0
                node_size_percentiles[node_id] = {'p10': 0, 'p50': 0, 'p90': 0}
            
            obs.extend([num_items, avg_size, avg_freq, avg_recency])
            
            free_space = node.get_free_space()
            cache_pressure = min(2.0, size / max(1, free_space))
            obs.append(cache_pressure / 2.0)  # Normalize to [0, 1]
            
            if self.include_static:
                obs.extend(self._static_features[node_id].tolist())
        
        obs.append(self._norm_size(size))
        obs.append(self._get_freq_hint(url))
        
        obs.append(self.total_hits / max(1, self.total_requests))
        
        total_tier_hits = sum(self.tier_hits.values()) or 1
        edge_ratio = self.tier_hits.get(0, 0) / total_tier_hits
        regional_ratio = self.tier_hits.get(1, 0) / total_tier_hits
        obs.extend([edge_ratio, regional_ratio])
        
        for node_id in self.nodes:
            node = self.topology.get_node(node_id)
            candidates = node.cache.get_candidates(self.k_candidates, timestamp)
            self._candidates[node_id] = candidates
            
            p10 = node_size_percentiles[node_id]['p10']
            p50 = node_size_percentiles[node_id]['p50']
            p90 = node_size_percentiles[node_id]['p90']
            
            for i in range(self.k_candidates):
                if i < len(candidates):
                    c = candidates[i]
                    obs.append(self._norm_size(c.size))
                    obs.append(self._norm_freq(c.frequency))
                    obs.append(np.exp(-0.001 * max(0, timestamp - c.last_access)))
                    
                    fit_ratio = min(2.0, size / max(1, c.size))
                    obs.append(fit_ratio / 2.0)  
                    
                    if c.size <= p10:
                        size_pct = 0.0
                    elif c.size <= p50:
                        size_pct = 0.25 + 0.25 * (c.size - p10) / max(1, p50 - p10)
                    elif c.size <= p90:
                        size_pct = 0.5 + 0.25 * (c.size - p50) / max(1, p90 - p50)
                    else:
                        size_pct = 0.75 + 0.25 * min(1.0, (c.size - p90) / max(1, p90))
                    obs.append(size_pct)
                else:
                    obs.extend([0.0, 0.0, 0.0, 1.0, 0.0])  
        
        return np.array(obs, dtype=np.float32)
    
    def _get_freq_hint(self, url: str) -> float:
        freq = self._url_freq.get(url, 0)
        return float(np.log1p(freq) / np.log1p(100))
    
    def _update_freq_history(self, url: str):
        self._url_freq[url] += 1
        self._url_history.append(url)
        
        while len(self._url_history) > self.freq_history_window:
            old_url = self._url_history.pop(0)
            self._url_freq[old_url] -= 1
            if self._url_freq[old_url] <= 0:
                del self._url_freq[old_url]
        
    def _apply(self, action: int) -> Tuple[float, bool, Dict]:
        url, size, timestamp, *_ = self.request
        info = {'processed': False, 'hit': False, 'size': size}
        evicted_item: Optional[CacheItem] = None
        
        lookup = self.topology.lookup(url, size, timestamp)
        self._last_lookup = lookup
        info['latency'] = lookup.total_latency_ms
        
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
        
        skip_action = self.num_nodes * self.k_candidates
        if action == skip_action:
            info['processed'] = True
            info['skipped'] = True
            return -0.05, False, info
        
        node_idx = action // self.k_candidates
        candidate_idx = action % self.k_candidates
        
        if node_idx >= self.num_nodes:
            info['processed'] = True
            info['skipped'] = True
            return -0.1, False, info
        
        node_id = self.nodes[node_idx]
        node = self.topology.get_node(node_id)
        
        if node.get_free_space() >= size:
            node.add(url, size, timestamp)
            info['processed'] = True
            info['cached_at'] = node_id
            return 0.0, False, info
        
        candidates = self._candidates.get(node_id, [])
        if not candidates or candidate_idx >= len(candidates):
            info['processed'] = True
            info['skipped'] = True
            return -0.15, False, info
        
        evicted_item = candidates[candidate_idx]
        node.evict(evicted_item.url)
        info['evicted'] = evicted_item.url
        info['evicted_size'] = evicted_item.size
        
        if node.get_free_space() >= size:
            node.add(url, size, timestamp)
            reward = self._reward(lookup, size, 1, [evicted_item])
            info['processed'] = True
            info['cached_at'] = node_id
            info['evicted_items'] = 1
            return reward, False, info
        else:
            info['processed'] = True
            info['need_more_eviction'] = True
            return -0.2, False, info
    
    def _reward(self, lookup: HierarchicalLookupResult, size: int, 
                evictions: int, evicted_items: Optional[List[CacheItem]] = None
                ) -> float:
        hit = lookup.hit and not lookup.from_origin
        url = self.request[0] if self.request else ""
        
        freq_importance = self._get_freq_hint(url)  
        size_importance = self._norm_size(size)      
        request_importance = 0.6 * freq_importance + 0.4 * size_importance
        
        alpha, beta, gamma = self._get_adaptive_weights()
        
        latency_reward = self._calc_latency_reward(lookup, size, hit)
        latency_reward *= (1.0 + request_importance)  
        
        tier_reward = self._calc_tier_reward(lookup, hit)
        
        eviction_penalty = self._calc_eviction_penalty(evictions, evicted_items)
        
        miss_penalty = self._calc_miss_penalty(hit, request_importance, size)
        
        reward = (alpha * latency_reward + 
                  beta * tier_reward - 
                  gamma * eviction_penalty -
                  miss_penalty)
        
        return reward
    
    def _get_adaptive_weights(self) -> Tuple[float, float, float]:
        if not self.adaptive_rewards:
            return self.alpha, self.beta, self.gamma
        
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        
        if self.total_requests < 10:
            return alpha, beta, gamma
        
        hit_rate = self.total_hits / self.total_requests
        if hit_rate < 0.3:
            alpha *= 1.3
        elif hit_rate > 0.7:
            beta *= 1.2
        
        total_tier_hits = sum(self.tier_hits.values()) or 1
        edge_ratio = self.tier_hits.get(0, 0) / total_tier_hits
        
        if edge_ratio < 0.2 and hit_rate > 0.3:
            beta *= 1.3
        elif edge_ratio > 0.8:
            beta *= 0.8
        
        avg_occupancy = np.mean([
            self.topology.get_node(n).get_occupancy() 
            for n in self.nodes
        ])
        if avg_occupancy > 0.9:
            gamma *= 1.5
        
        total = alpha + beta + gamma
        scale = (self.alpha + self.beta + self.gamma) / total
        
        return alpha * scale, beta * scale, gamma * scale
    
    def _calc_latency_reward(self, lookup: HierarchicalLookupResult, 
                              size: int, hit: bool) -> float:
        if not hit:
            return 0.0
        
        max_latency = self.topology.calculate_origin_fetch_time(size)
        savings = (max_latency - lookup.total_latency_ms) / max_latency
        return max(0.0, savings)
    
    def _calc_tier_reward(self, lookup: HierarchicalLookupResult, 
                          hit: bool) -> float:
        if not hit or lookup.hit_tier is None:
            return 0.0
        
        max_tier = max(1, self._num_tiers)
        base_bonus = 1.0 - (lookup.hit_tier / max_tier)
        
        hit_node = lookup.hit_node_id
        if hit_node:
            node = self.topology.get_node(hit_node)
            occupancy = node.get_occupancy()
            pressure_multiplier = 0.7 + 0.6 * occupancy  
            base_bonus *= pressure_multiplier
        
        return base_bonus
    
    def _calc_eviction_penalty(self, evictions: int, 
                                evicted_items: Optional[List[CacheItem]] = None
                                ) -> float:
        if evictions == 0:
            return 0.0
        
        base_penalty = evictions / max(1, self.max_evictions)
        
        if evicted_items:
            timestamp = self.request[2] if self.request else 0
            total_value = 0.0
            
            for item in evicted_items:
                freq_value = self._norm_freq(item.frequency)
                
                recency_value = np.exp(-0.001 * max(0, timestamp - item.last_access))
                
                size_value = self._norm_size(item.size)
                
                item_value = 0.4 * freq_value + 0.4 * recency_value + 0.2 * size_value
                total_value += item_value
            
            avg_value = total_value / len(evicted_items)
            value_multiplier = 0.5 + avg_value
            base_penalty *= value_multiplier
        
        return min(1.0, base_penalty)
    
    def _calc_miss_penalty(self, hit: bool, request_importance: float,
                           size: int) -> float:
        if hit:
            return 0.0
        base_penalty = 0.3
        importance_scale = 0.5 + request_importance
        
        if size > 1_000_000:  
            importance_scale *= 1.2
        
        return base_penalty * importance_scale
    
    
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
            'steps': self.step_count
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
