import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy
from src.agents.dqn import PlacementDQN

def main():
    parser = argparse.ArgumentParser(description='Train Placement DQN Agent')
    parser.add_argument('--steps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--model_name', type=str, default='dqn_baseline', help='Name of the model to save')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=1000, help='Steps before learning starts')
    parser.add_argument('--batch-size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--exploration-fraction', type=float, default=0.1, help='Fraction of steps for exploration')
    parser.add_argument('--exploration-final-eps', type=float, default=0.05, help='Final exploration probability')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size (default: 64)')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    os.environ['PYTHONUNBUFFERED'] = '1'

    data_path = 'data/processed/consistent_trace.csv'
    model_path = f"models/{args.model_name}"
    
    loader = DataLoader(data_path, split_ratio=0.7, mode='train')
    topology = create_simple_hierarchy(
        edge_capacity_mb=1.0,
        regional_capacity_mb=2.0,
        origin_latency_ms=100.0
    )
    env = CacheEnv(data_loader=loader, topology=topology)
    
    policy_kwargs = dict(net_arch=[args.hidden_size, args.hidden_size])
    
    agent = PlacementDQN(
        env=env,
        model_path=f"models/{args.model_name}",
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=1000,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    agent.train(total_timesteps=args.steps, log_interval=1000)
    agent.save()

if __name__ == "__main__":
    main()
