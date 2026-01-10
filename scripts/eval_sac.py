import sys
import os
import argparse
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy
from src.agents.sac_discrete import DiscreteSACAgent

ACTION_NAMES = {
    0: "Cache at Edge",
    1: "Cache at Regional",
    2: "Skip (No Cache)"
}


def get_cache_stats(env) -> Dict:
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    topology = unwrapped.topology
    
    stats = {}
    for node_id in unwrapped.nodes:
        node = topology.get_node(node_id)
        tier = topology._configs[node_id].tier
        stats[node_id] = {
            'tier': tier,
            'occupancy': node.get_occupancy(),
            'used_bytes': node.cache.current_size,
            'capacity_bytes': node.cache.capacity,
            'num_items': len(node.cache),
        }
    return stats


def evaluate_agent(agent, env, num_steps: int, verbose: bool = True) -> Dict:
    obs, _ = env.reset()
    
    total_reward = 0.0
    action_counts = defaultdict(int)
    rewards_per_action = defaultdict(list)
    step_rewards = []
    cache_occupancy_history = []
    
    for step in range(1, num_steps + 1):
        action = agent.select_action(obs, evaluate=True)
        action_counts[action] += 1
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_rewards.append(reward)
        rewards_per_action[action].append(reward)
        
        if step % 100 == 0:
            cache_occupancy_history.append(get_cache_stats(env))
        
        obs = next_obs
        
        if verbose and step % 1000 == 0:
            unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
            metrics = unwrapped.get_metrics()
            print(f"Step {step:>6}/{num_steps} | "
                  f"Reward: {total_reward:>8.2f} | "
                  f"Hit Rate: {metrics['hit_rate']:.2%} | "
                  f"Byte Hit: {metrics['byte_hit_rate']:.2%}")
        
        if terminated:
            obs, _ = env.reset()
    
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    return {
        'total_reward': total_reward,
        'mean_reward': float(np.mean(step_rewards)),
        'std_reward': float(np.std(step_rewards)),
        'action_counts': dict(action_counts),
        'rewards_per_action': {
            k: (float(np.mean(v)), float(np.std(v))) 
            for k, v in rewards_per_action.items()
        },
        'cache_history': cache_occupancy_history,
        'final_metrics': unwrapped.get_metrics()
    }


def print_cache_summary(cache_history: List[Dict]):
    if not cache_history:
        return
    
    final = cache_history[-1]
    
    print("Cache Occupancy:")
    
    for node_id, stats in sorted(final.items(), key=lambda x: x[1]['tier']):
        tier_name = "Edge" if stats['tier'] == 0 else f"Tier {stats['tier']}"
        used_mb = stats['used_bytes'] / (1024 * 1024)
        capacity_mb = stats['capacity_bytes'] / (1024 * 1024)
        
        print(f"\n{node_id} ({tier_name}):")
        print(f"Occupancy:    {stats['occupancy']:6.1%}")
        print(f"Used:         {used_mb:6.2f} MB / {capacity_mb:.2f} MB")
        print(f"Cached Items: {stats['num_items']:6d}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Discrete SAC Agent')
    
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    
    parser.add_argument('--cache-mb', type=float, default=1.0, help='Edge cache size in MB')
    parser.add_argument('--regional-mb', type=float, default=2.0, help='Regional cache size in MB')
    parser.add_argument('--data', type=str, default='data/processed/consistent_trace.csv')
    
    parser.add_argument('--steps', type=int, default=10000, help='Evaluation steps')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Discrete SAC Evaluation:\n")
    print(f"Model:       {args.model}")
    print(f"Data Mode:   {args.mode}")
    print(f"Steps:       {args.steps:,}")
    print(f"Cache:       Edge={args.cache_mb}MB, Regional={args.regional_mb}MB")
    print(f"Device:      {device}")
    
    loader = DataLoader(args.data, split_ratio=0.7, mode=args.mode)
    topology = create_simple_hierarchy(
        edge_capacity_mb=args.cache_mb,
        regional_capacity_mb=args.regional_mb
    )
    env = CacheEnv(loader, topology=topology)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DiscreteSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    agent.load(args.model)
    
    results = evaluate_agent(agent, env, num_steps=args.steps, verbose=not args.quiet)
    
    metrics = results['final_metrics']

    print(f"\nTotal Reward:     {results['total_reward']:.2f}")
    print(f"Mean Step Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"\nHit Rate:         {metrics['hit_rate']:.2%}")
    print(f"Byte Hit Rate:    {metrics['byte_hit_rate']:.2%}")
    print(f"Avg Latency:      {metrics['avg_latency']:.2f} ms")
    print(f"Total Requests:   {metrics['requests']:,}")
    print(f"Total Hits:       {metrics['hits']:,}")
    
    print("\n Action Distribution:")
    
    total_actions = sum(results['action_counts'].values())
    for action_id in sorted(results['action_counts'].keys()):
        count = results['action_counts'][action_id]
        pct = count / total_actions
        mean_r, std_r = results['rewards_per_action'][action_id]
        name = ACTION_NAMES.get(action_id, f"Action {action_id}")
        print(f"  {name:20}: {count:>6,} ({pct:5.1%}) | Reward: {mean_r:+.4f} ± {std_r:.4f}")
    
    if total_actions > 0:
        skip_pct = results['action_counts'].get(2, 0) / total_actions
        regional_pct = results['action_counts'].get(1, 0) / total_actions
        
    print_cache_summary(results['cache_history'])
    
    return 0 if metrics['hit_rate'] > 0.1 else 1


if __name__ == "__main__":
    exit(main())
