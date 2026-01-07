import argparse
import time
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.cache_env import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy
from src.agents.ppo import PlacementPPO
from src.data_loader import DataLoader

def evaluate_policy(env, policy, name, steps=1000):
    obs, _ = env.reset()
    total_reward = 0
    latencies = []
    actions = []
    
    start_time = time.time()
    
    for _ in range(steps):
        if policy:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample() # Random
            
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        actions.append(action)
        
        if 'latency' in info:
            latencies.append(info['latency'])
            
        if done:
            obs, _ = env.reset()
            
    duration = time.time() - start_time
    metrics = env.get_metrics()
    
    unique, counts = np.unique(actions, return_counts=True)
    action_dist = dict(zip(unique, counts))
    
    return {
        'hit_rate': metrics['hit_rate'],
        'byte_hit_rate': metrics['byte_hit_rate'],
        'avg_latency': np.mean(latencies) if latencies else 0.0,
        'reward': total_reward,
        'action_counts': action_dist,
        'time': duration
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO Placement Agent')
    parser.add_argument('--model', type=str, default='models/ppo_model', help='Path to model')
    parser.add_argument('--steps', type=int, default=10000, help='Number of evaluation steps')
    args = parser.parse_args()

    data_path = 'data/processed/consistent_trace.csv'
    loader = DataLoader(data_path, split_ratio=0.7, mode='test')
    
    topology = create_simple_hierarchy(
        edge_capacity_mb=1.0,
        regional_capacity_mb=2.0,
        origin_latency_ms=100.0
    )
    env = CacheEnv(data_loader=loader, topology=topology)
    
    print(f"\nEvaluating PPO for {args.steps} steps...")
    
    ppo_agent = None
    try:
        ppo_agent = PlacementPPO.load(args.model, env=env)
    except FileNotFoundError:
        print(f"Model not found at {args.model}. Exiting.")
        sys.exit(1)
    
    ppo_metrics = evaluate_policy(env, ppo_agent, "PlacementPPO", args.steps)
    
    print("\nResults:")
    print(f"  Hit Rate:      {ppo_metrics['hit_rate']*100:.2f}%")
    print(f"  Byte Hit Rate: {ppo_metrics['byte_hit_rate']*100:.2f}%")
    print(f"  Avg Latency:   {ppo_metrics['avg_latency']:.2f} ms")
    print(f"  Total Reward:  {ppo_metrics['reward']:.2f}")
    
    actions = ppo_metrics.get('action_counts', {})
    total_actions = sum(actions.values())
    if total_actions > 0:
        dist_str = ", ".join([f"{k}: {v/total_actions*100:.1f}%" for k, v in sorted(actions.items())])
        print(f"  Actions:       {dist_str}")
    
    print(f"  Time:          {ppo_metrics['time']:.2f}s")

if __name__ == "__main__":
    main()
