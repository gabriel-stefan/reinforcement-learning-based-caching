import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SoftQNetwork(nn.Module):
    def __init__(self, envs, hidden_size=256):
        super().__init__()
        if hasattr(envs, 'single_observation_space'):
            obs_shape = envs.single_observation_space.shape
            action_dim = envs.single_action_space.n
        else:
            obs_shape = envs.observation_space.shape
            action_dim = envs.action_space.n
            
        self.net = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, action_dim)),
        )

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, envs, hidden_size=256):
        super().__init__()
        if hasattr(envs, 'single_observation_space'):
            obs_shape = envs.single_observation_space.shape
            action_dim = envs.single_action_space.n
        else:
            obs_shape = envs.observation_space.shape
            action_dim = envs.action_space.n

        self.net = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, action_dim)),
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


def analyze_decision_making(model_path, steps=500000, output_dir='plots/sac_analysis'):
    print(f"Analyzing Discrete SAC for {steps:,} steps...")
    
    data_path = 'data/processed/consistent_trace.csv'
    loader = DataLoader(data_path, split_ratio=0.7, mode='test')
    topology = create_simple_hierarchy()
    env = CacheEnv(data_loader=loader, topology=topology)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    hidden_size = checkpoint.get('hidden_size', 256)
    actor = Actor(env, hidden_size).to(device)
    qf1 = SoftQNetwork(env, hidden_size).to(device)
    qf2 = SoftQNetwork(env, hidden_size).to(device)
    
    actor.load_state_dict(checkpoint['actor_state_dict'])
    qf1.load_state_dict(checkpoint['qf1_state_dict'])
    qf2.load_state_dict(checkpoint['qf2_state_dict'])
    
    actor.eval()
    qf1.eval()
    qf2.eval()
    
    sample_rate = max(1, steps // 10000)  # 10k points
    n_samples = steps // sample_rate + 1
    
    probs_arr = np.zeros((n_samples, 3), dtype=np.float32)
    q_values_arr = np.zeros((n_samples, 3), dtype=np.float32)
    entropy_arr = np.zeros(n_samples, dtype=np.float32)
    actions_arr = np.zeros(n_samples, dtype=np.int8)
    rewards_arr = np.zeros(steps, dtype=np.float32)
    
    sample_idx = 0
    obs, _ = env.reset()
    
    for i in tqdm(range(steps), desc="Evaluating", unit="step", miniters=1000):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(device)
            
            _, _, probs = actor.get_action(obs_tensor)
            probs_np = probs.cpu().numpy()[0]
            dist = Categorical(probs=probs)
            entropy = dist.entropy().cpu().item()
            
            q1 = qf1(obs_tensor)
            q2 = qf2(obs_tensor)
            q_min = torch.min(q1, q2) # Conservative Q-value
            q_values_np = q_min.cpu().numpy()[0]
        
        action = np.argmax(probs_np)
        
        if i % sample_rate == 0 and sample_idx < n_samples:
            probs_arr[sample_idx] = probs_np
            q_values_arr[sample_idx] = q_values_np
            entropy_arr[sample_idx] = entropy
            actions_arr[sample_idx] = action
            sample_idx += 1
        
        obs, reward, done, truncated, info = env.step(action)
        rewards_arr[i] = reward
        
        if done or truncated:
            obs, _ = env.reset()
    
    probs_arr = probs_arr[:sample_idx]
    q_values_arr = q_values_arr[:sample_idx]
    entropy_arr = entropy_arr[:sample_idx]
    actions_arr = actions_arr[:sample_idx]
    
    probs_df = pd.DataFrame(probs_arr, columns=['Edge', 'Regional', 'Skip'])
    q_df = pd.DataFrame(q_values_arr, columns=['Edge', 'Regional', 'Skip'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    
    steps_x = np.arange(len(probs_df)) * sample_rate
    
    plt.figure(figsize=(10, 6))
    window = 100
    for col, color in zip(['Edge', 'Regional', 'Skip'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        smooth = probs_df[col].rolling(window=window, min_periods=1).mean()
        plt.plot(steps_x, smooth, label=col, color=color, linewidth=2)
        plt.fill_between(steps_x, smooth, alpha=0.1, color=color)
        
    plt.title('Action Probability Evolution')
    plt.xlabel('Step')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/action_prob_evolution.png', dpi=150)
    plt.close()
    print("Generated action_prob_evolution.png")

    plt.figure(figsize=(10, 6))
    window = 100
    for col, color in zip(['Edge', 'Regional', 'Skip'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        smooth = q_df[col].rolling(window=window, min_periods=1).mean()
        plt.plot(steps_x, smooth, label=col, color=color, linewidth=2)
        plt.fill_between(steps_x, smooth, alpha=0.1, color=color)
        
    plt.title('Q-Value Evolution (Min Q1, Q2)')
    plt.xlabel('Step')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/q_value_evolution.png', dpi=150)
    plt.close()
    print("Generated q_value_evolution.png")
    
    plt.figure(figsize=(10, 6))
    window = 1000
    ent_smooth = pd.Series(entropy_arr).rolling(window=window, min_periods=1).mean()
    plt.plot(steps_x, ent_smooth, color='purple', linewidth=2)
    plt.title('Policy Entropy Evolution')
    plt.xlabel('Step')
    plt.ylabel('Entropy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_evolution.png', dpi=150)
    plt.close()
    print("Generated entropy_evolution.png")
    
    plt.figure(figsize=(10, 6))
    window = 5000
    reward_smooth = pd.Series(rewards_arr).rolling(window=window, min_periods=1).mean()
    plt.plot(reward_smooth, color='green', linewidth=1.5)
    plt.title(f'Rolling Average Reward (window={window})')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_average_reward.png', dpi=150)
    plt.close()
    print("Generated rolling_average_reward.png")

    plt.figure(figsize=(10, 6))
    action_dummies = pd.get_dummies(actions_arr)
    for i in range(3):
        if i not in action_dummies.columns:
            action_dummies[i] = 0
    action_dummies = action_dummies[[0, 1, 2]]

    window = 1000
    action_probs = action_dummies.rolling(window=window, min_periods=1).mean()
    
    plt.stackplot(steps_x, 
                  action_probs[0], action_probs[1], action_probs[2],
                  labels=['Edge', 'Regional', 'Skip'],
                  colors=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    
    plt.title('Action Distribution Evolution')
    plt.xlabel('Step')
    plt.ylabel('Probability')
    plt.legend(loc='upper left')
    plt.margins(0, 0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/action_distribution_evolution.png', dpi=150)
    plt.close()
    print("Generated action_distribution_evolution.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SAC Deep Dive Minimal')
    parser.add_argument('--steps', type=int, default=500000, help='Number of evaluation steps')
    parser.add_argument('--model', type=str, default='models/sac15_stable.pt', help='Path to model (.pt file)')
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        analyze_decision_making(args.model, steps=args.steps)
    else:
        print(f"Model {args.model} not found.")
