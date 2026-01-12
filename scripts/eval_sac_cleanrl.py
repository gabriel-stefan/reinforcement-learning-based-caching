import os
import sys

sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, envs, hidden_size=256):
        super().__init__()
        if hasattr(envs, 'single_observation_space'):
            obs_shape = envs.single_observation_space.shape
            action_dim = envs.single_action_space.n
        else:
            obs_shape = envs.observation_space.shape
            action_dim = envs.action_space.n

        self.fc1 = layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc3 = layer_init(nn.Linear(hidden_size, action_dim))

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action(self, x):
        logits = self(x)
        action = torch.argmax(logits, dim=1)
        return action

def evaluate(model_path, data_path, num_steps, device, hidden_size=256):
    loader = DataLoader(data_path, split_ratio=0.7, mode='test')
    topology = create_simple_hierarchy(
        edge_capacity_mb=1.0,
        regional_capacity_mb=2.0,
        origin_latency_ms=100.0
    )
    env = CacheEnv(data_loader=loader, topology=topology)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    checkpoint = torch.load(model_path, map_location=device)
    
    saved_hidden_size = checkpoint.get('hidden_size', 256)
    if hidden_size == 256 and saved_hidden_size != 256:
        print(f"Using hidden size {saved_hidden_size} from checkpoint.", flush=True)
        hidden_size = saved_hidden_size
    
    actor = Actor(env, hidden_size).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    print(f"Loaded model from {model_path} (hidden_size={hidden_size})", flush=True)

    obs, _ = env.reset()
    total_reward = 0
    action_counts = defaultdict(int)
    
    print(f"Starting evaluation for {num_steps} steps", flush=True)
    
    for step in range(num_steps):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            action = actor.get_action(obs_tensor)
            action = action.cpu().numpy()[0]

        action_counts[action] += 1
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 1000 == 0:
            metrics = env.unwrapped.get_metrics()
            print(f"Step {step}: Hit Rate={metrics['hit_rate']:.2%}, Byte Hit Rate={metrics['byte_hit_rate']:.2%}", flush=True)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # Final Metrics
    metrics = env.unwrapped.get_metrics()
    print("\n=== Evaluation Results ===")
    print(f"Total Steps: {num_steps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Hit Rate: {metrics['hit_rate']:.2%}")
    print(f"Byte Hit Rate: {metrics['byte_hit_rate']:.2%}")
    print(f"Average Latency: {metrics['avg_latency']:.2f} ms")
    
    print("\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        print(f"Action {action}: {count} ({count/total_actions:.1%})")

if __name__ == "__main__":
    print("Parsing arguments...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/processed/consistent_trace.csv", help="Path to data file")
    parser.add_argument("--steps", type=int, default=10000, help="Number of evaluation steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    
    args = parser.parse_args()
    
    print(f"Arguments parsed. Device: {args.device}", flush=True)
    evaluate(args.model, args.data, args.steps, args.device, args.hidden_size)
