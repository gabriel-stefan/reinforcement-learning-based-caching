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
    args = parser.parse_args()

    data_path = 'data/processed/consistent_trace.csv'
    loader = DataLoader(data_path, split_ratio=0.7, mode='train')
    
    topology = create_simple_hierarchy(
        edge_capacity_mb=1.0,
        regional_capacity_mb=2.0,
        origin_latency_ms=100.0
    )
    env = CacheEnv(data_loader=loader, topology=topology)
    
    print("Initializing PPO Agent...")
    agent = PlacementPPO(
        env=env,
        model_path=f"models/{args.model_name}",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0
    )

    agent.train(total_timesteps=args.steps, log_interval=1000)
    
    agent.save()

if __name__ == "__main__":
    main()
