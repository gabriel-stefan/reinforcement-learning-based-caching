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
    parser.add_argument('--steps', type=int, default=200000, help='Total training timesteps')
    parser.add_argument('--model_name', type=str, default='dqn', help='Name of the model to save')
    args = parser.parse_args()

    data_path = 'data/processed/consistent_trace.csv'
    model_path = f"models/{args.model_name}"
    
    loader = DataLoader(data_path, split_ratio=0.7, mode='train')
    topology = create_simple_hierarchy(
        edge_capacity_mb=1.0,
        regional_capacity_mb=2.0,
        origin_latency_ms=100.0
    )
    env = CacheEnv(data_loader=loader, topology=topology)
    agent = PlacementDQN(
        env=env,
        model_path=f"models/{args.model_name}",
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=0
    )

    agent.train(total_timesteps=args.steps, log_interval=1000)
    agent.save()

if __name__ == "__main__":
    main()
