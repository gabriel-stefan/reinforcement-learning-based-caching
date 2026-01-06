import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy
from src.agents.dqn import PlacementDQN
from src.agents.cdn_baselines import EdgeFirstLFU, SizeSplitLFU

def evaluate_policy(env, policy, name, steps=10000):
    print(f"\nEvaluating {name} for {steps} steps...")
    obs, _ = env.reset()
    start_time = time.time()
    
    total_reward = 0.0
    latencies = []
    actions = []
    
    for i in range(steps):
        if hasattr(policy, 'predict'):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy(obs)
            
        obs, reward, done, _, info = env.step(action)
        
        total_reward += reward
        actions.append(action)
        if 'latency' in info:
            latencies.append(info['latency'])
            
        if done:
            break
            
    duration = time.time() - start_time
    metrics = env.get_metrics()
    
    # action distribution
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
    parser = argparse.ArgumentParser(description='Evaluate Placement DQN Agent')
    parser.add_argument('--model', type=str, default='models/placement_dqn_v1', help='Path to trained model')
    parser.add_argument('--steps', type=int, default=20000, help='Evaluation steps')
    args = parser.parse_args()

    data_path = 'data/processed/consistent_trace.csv'
    loader = DataLoader(data_path, split_ratio=0.7, mode='test')
    
    topology = create_simple_hierarchy(
        edge_capacity_mb=1.0,
        regional_capacity_mb=2.0,
        origin_latency_ms=100.0
    )
    env = CacheEnv(data_loader=loader, topology=topology)
    
    print(f"\nEvaluating DQN for {args.steps} steps...")
    
    dqn_agent = None
    try:
        dqn_agent = PlacementDQN.load(args.model, env=env)
    except FileNotFoundError:
        print(f"Model not found at {args.model}!")
        sys.exit(1) 
    
    dqn_metrics = evaluate_policy(env, dqn_agent, "DQN", args.steps)
    
    print("\nResults:")
    print(f"  Hit Rate:      {dqn_metrics['hit_rate']*100:.2f}%")
    print(f"  Byte Hit Rate: {dqn_metrics['byte_hit_rate']*100:.2f}%")
    print(f"  Avg Latency:   {dqn_metrics['avg_latency']:.2f} ms")
    print(f"  Total Reward:  {dqn_metrics['reward']:.2f}")
    
    actions = dqn_metrics.get('action_counts', {})
    total_actions = sum(actions.values())
    if total_actions > 0:
        dist_str = ", ".join([f"{k}: {v/total_actions*100:.1f}%" for k, v in sorted(actions.items())])
        print(f"  Actions:       {dist_str}")
    
    print(f"  Time:          {dqn_metrics['time']:.2f}s")

if __name__ == "__main__":
    main()
