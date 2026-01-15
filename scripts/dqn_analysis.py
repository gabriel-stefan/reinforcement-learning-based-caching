import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import sys
from tqdm import tqdm
from stable_baselines3 import DQN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy

def analyze_decision_making(model_path, steps=500000, output_dir='plots/dqn_analysis'):
    print(f"Analyzing DQNak for {steps:,} steps")
    
    data_path = 'data/processed/consistent_trace.csv'
    loader = DataLoader(data_path, split_ratio=0.7, mode='test')
    topology = create_simple_hierarchy()
    env = CacheEnv(data_loader=loader, topology=topology)
    
    model = DQN.load(model_path, env=env)
    sample_rate = max(1, steps // 10000)  
    n_samples = steps // sample_rate + 1

    q_values_arr = np.zeros((n_samples, 3), dtype=np.float32)
    actions_arr = np.zeros(n_samples, dtype=np.int8)
    rewards_arr = np.zeros(steps, dtype=np.float32)
    q_gaps_arr = np.zeros(n_samples, dtype=np.float32)
    
    
    sample_idx = 0
    obs, _ = env.reset()
    
    for i in tqdm(range(steps), desc="Evaluating", unit="step", miniters=1000):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
            q_values = model.policy.q_net(obs_tensor)
            q_values_np = q_values.cpu().numpy()[0]

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        if i % sample_rate == 0 and sample_idx < n_samples:
            sorted_q = np.sort(q_values_np)[::-1]
            q_gaps_arr[sample_idx] = sorted_q[0] - sorted_q[1]
            q_values_arr[sample_idx] = q_values_np
            actions_arr[sample_idx] = action
            sample_idx += 1
        
        obs, reward, done, truncated, info = env.step(action)
        rewards_arr[i] = reward
 
        
        if done or truncated:
            obs, _ = env.reset()

    q_values_arr = q_values_arr[:sample_idx]
    actions_arr = actions_arr[:sample_idx]
    q_gaps_arr = q_gaps_arr[:sample_idx]

    q_df = pd.DataFrame(q_values_arr, columns=['Edge', 'Regional', 'Skip'])
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

    plt.figure(figsize=(10, 6))
    window = 100
    steps_x = np.arange(len(q_df)) * sample_rate
    
    for col, color in zip(['Edge', 'Regional', 'Skip'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        smooth = q_df[col].rolling(window=window, min_periods=1).mean()
        plt.plot(steps_x, smooth, label=col, color=color, linewidth=2)
        plt.fill_between(steps_x, smooth, alpha=0.1, color=color)
        
    plt.title('Q-Value Evolution')
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
    gap_smooth = pd.Series(q_gaps_arr).rolling(window=window, min_periods=1).mean()
    plt.plot(steps_x, gap_smooth, color='purple', linewidth=2)
    plt.fill_between(steps_x, gap_smooth, alpha=0.2, color='purple')
    plt.title('Q-Value Confidence Gap')
    plt.xlabel('Step')
    plt.ylabel('Gap (Best - 2nd Best)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/q_value_confidence_gap.png', dpi=150)
    plt.close()
    print("Generated q_value_confidence_gap.png")
    
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
    parser = argparse.ArgumentParser(description='DQN Deep Dive Minimal')
    parser.add_argument('--steps', type=int, default=500000, help='Number of evaluation steps')
    parser.add_argument('--model', type=str, default='models/dqn_batch_128', help='Path to model (without .zip)')
    args = parser.parse_args()
    
    if os.path.exists(args.model + ".zip"):
        analyze_decision_making(args.model, steps=args.steps)
    else:
        print(f"Model {args.model} not found.")
