import argparse
import time
import numpy as np
from src.data_loader import DataLoader
from src.environments.core.network_topology import create_simple_hierarchy
from src.agents.cdn_baselines import edge_first, size_split, probabilistic, percentile_split

def run_simulation(loader, policy, name, steps):
    print(f"evaluating {name}...")
    
    # setup topology (edge=1mb, regional=2mb)
    topology = create_simple_hierarchy(edge_capacity_mb=1.0, regional_capacity_mb=2.0)
    edge_node = topology.get_node(topology.get_cache_node_ids()[0])
    regional_node = topology.get_node(topology.get_cache_node_ids()[1])
    
    loader.reset()
    total_reward = 0
    total_hits = 0
    start_time = time.time()
    max_size_norm = 5_000_000
    
    for i in range(steps):
        request = loader.get_next()
        if request is None: break
            
        url, size, timestamp, *_ = request
        
        # minimal observation for policy (size at index -5)
        dummy_obs = [0] * 10
        dummy_obs[-5] = np.log1p(size) / np.log1p(max_size_norm)
        
        action, _ = policy.predict(dummy_obs)
        
        # 1. lookup
        lookup = topology.lookup(url, size, timestamp)
        if lookup.hit and not lookup.from_origin:
            total_reward += 10.0 if lookup.hit_tier == 0 else 8.0
            total_hits += 1
            continue
            
        # 2. miss - placement
        if action == 2: 
            total_reward -= 1.0
            continue
            
        target = edge_node if action == 0 else regional_node
        
        # 3. eviction (lfu)
        if target.get_free_space() < size:
            items = target.cache.get_all_items()
            if items:
                for _ in range(20): # max 20 evictions
                    victim = min(items, key=lambda x: x.frequency)
                    target.evict(victim.url)
                    if target.get_free_space() >= size: break
                    items = target.cache.get_all_items()
        
        # 4. cache
        if target.get_free_space() >= size:
            target.add(url, size, timestamp)
            r = -0.5
            if action == 0: # Edge Tax
                r -= (size / target.cache.capacity) * 5.0
            total_reward += r
        else:
            total_reward -= 1.0 # miss (failed to cache)

    elapsed = time.time() - start_time
    hit_rate = total_hits / (i + 1)
    
    print(f"  hit rate: {hit_rate:.2%}, reward: {total_reward:.2f}, time: {elapsed:.1f}s")
    return {'hit_rate': hit_rate, 'reward': total_reward, 'time': elapsed}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000000)
    args = parser.parse_args()
    
    print(f"--- cdn placement benchmark (steps={args.steps}) ---")
    loader = DataLoader('data/processed/consistent_trace.csv', split_ratio=0.7, mode='test')
    
    policies = [
        ('edge_first', edge_first()),
        ('size_split_median', size_split(threshold=2463)),
        ('probabilistic', probabilistic(p=0.1)),
        ('percentile_split', percentile_split(threshold=4299))
    ]
    
    results = []
    for name, policy in policies:
        results.append((name, run_simulation(loader, policy, name, args.steps)))
        
    print(f"\n{'policy':<20} {'hit rate':>10} {'reward':>10} {'time':>8}")
    print("-" * 52)
    for name, m in results:
        print(f"{name:<20} {m['hit_rate']:>9.2%} {m['reward']:>10.2f} {m['time']:>7.1f}s")

if __name__ == "__main__":
    main()
