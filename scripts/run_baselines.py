import argparse
import time
import numpy as np
from src.data_loader import DataLoader
from src.environments.core.network_topology import create_simple_hierarchy
from src.environments.core.cache_storage import CacheStorage
from src.agents.baselines import get_policy, EvictionPolicy

def evaluate_policy(loader: DataLoader, eviction_policy: EvictionPolicy, 
                   total_capacity: int, steps: int, verbose: bool = True) -> dict:
   #  simulating a unified cache with the given eviction policy

    if verbose:
        print(f"Evaluating {eviction_policy.name.upper()}...")
    
    loader.reset()
    cache = CacheStorage(capacity_bytes=total_capacity)
    total_requests = 0
    total_hits = 0
    hit_bytes = 0
    total_bytes = 0
    start_time = time.time()
    
    for _ in range(steps):
        request = loader.get_next()
        if request is None:
            break
            
        url, size, timestamp, *_ = request
        total_requests += 1
        total_bytes += size
        
        if cache.contains(url):
            total_hits += 1
            hit_bytes += size
            cache.access(url, timestamp)
            continue
        
        if cache.get_free_space() >= size:
            cache.add(url, size, timestamp)
        else:
            items = cache.get_all_items()
            if items:
                for _ in range(20):
                    victim = eviction_policy.select_victim(items, timestamp)
                    if victim is None:
                        break
                        
                    cache.evict(victim.url)
                    
                    if cache.get_free_space() >= size:
                        cache.add(url, size, timestamp)
                        break
                    items = cache.get_all_items()

    elapsed_time = time.time() - start_time
    
    return {
        'hit_rate': total_hits / max(1, total_requests),
        'byte_hit_rate': hit_bytes / max(1, total_bytes),
        'elapsed_time': elapsed_time,
        'steps': total_requests
    }

def print_results(results):
    print(f"\n{'Policy':<12} {'Hit Rate':>10} {'Byte Hit':>10} {'Time':>8}")
    print("-" * 42)
    
    for name, m in results:
        print(f"{name:<12} {m['hit_rate']:>9.2%} {m['byte_hit_rate']:>9.2%} {m['elapsed_time']:>7.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Cache Eviction Policies")
    parser.add_argument('--steps', type=int, default=1000000, help="Number of simulation steps")
    parser.add_argument('--cache-mb', type=float, default=1.0, help="Edge capacity in MB")
    parser.add_argument('--regional-mb', type=float, default=2.0, help="Regional capacity in MB")
    parser.add_argument('--policies', nargs='+', 
                        default=['lru', 'lfu', 'fifo', 'random'],
                        choices=['random', 'lru', 'lfu', 'fifo', 'size'],
                        help="Policies to benchmark")
    parser.add_argument('--data', type=str, default='data/processed/consistent_trace.csv', help="Path to trace file")
    parser.add_argument('--quiet', action='store_true', help="Suppress progress output")
    
    args = parser.parse_args()
    
    edge_bytes = int(args.cache_mb * 1024 * 1024)
    regional_bytes = int(args.regional_mb * 1024 * 1024)
    total_capacity = edge_bytes + regional_bytes
    
    loader = DataLoader(args.data, split_ratio=0.7, mode='test')
    results = []
    
    for policy_name in args.policies:
        policy = get_policy(policy_name)
        metrics = evaluate_policy(loader, policy, total_capacity, args.steps, not args.quiet)
        results.append((policy_name, metrics))
        
    print_results(results)

if __name__ == "__main__":
    main()
