import argparse
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.cache_env import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy
from src.agents.ppo import PlacementPPO
from src.data_loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Train PPO Placement Agent')
    parser.add_argument('--steps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--model_name', type=str, default='ppo_model', help='Name of model to save')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048, help='Number of steps to run for each environment per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epoch when optimizing the surrogate loss')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Factor for trade-off of bias vs variance for GAE')
    parser.add_argument('--clip-range', type=float, default=0.2, help='Clipping parameter')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size (default: 64)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    os.environ['PYTHONUNBUFFERED'] = '1'

    data_path = 'data/processed/consistent_trace.csv'
    loader = DataLoader(data_path, split_ratio=0.7, mode='train')
    
    topology = create_simple_hierarchy(
        edge_capacity_mb=1.0,
        regional_capacity_mb=2.0,
        origin_latency_ms=100.0
    )
    env = CacheEnv(data_loader=loader, topology=topology)
    
    print(f"Initializing PPO Agent with seed {args.seed}...")
    
    # Define network architecture
    policy_kwargs = dict(net_arch=[dict(pi=[args.hidden_size, args.hidden_size], vf=[args.hidden_size, args.hidden_size])])
    
    agent = PlacementPPO(
        env=env,
        model_path=f"models/{args.model_name}",
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    agent.train(total_timesteps=args.steps, log_interval=1000)
    
    agent.save()

if __name__ == "__main__":
    main()
