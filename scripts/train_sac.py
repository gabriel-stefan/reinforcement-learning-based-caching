import sys
import os
import argparse
import numpy as np
import torch
import time
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy
from src.agents.sac_discrete import DiscreteSACAgent
import gymnasium as gym

ACTION_NAMES = {0: 'Edge', 1: 'Regional', 2: 'Skip'}


class EpisodeWrapper(gym.Wrapper): #wrapper that splits the continuous data stream into fixed-length episodes
    
    def __init__(self, env, max_steps_per_episode=5000):
        super().__init__(env)
        self.max_steps = max_steps_per_episode
        self.current_step = 0
    
    def reset(self, **kwargs):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.unwrapped._get_obs()
            info = {'continued_episode': True}
        
        self.current_step = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            truncated = True
            self.current_step = 0
        
        if terminated:
            self.current_step = 0
        
        return obs, reward, terminated, truncated, info


def evaluate(agent, env, num_episodes=5):
    total_rewards = []
    hit_rates = []
    byte_hit_rates = []
    action_counts = defaultdict(int)
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(obs, evaluate=True)
            action_counts[action] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        metrics = env.unwrapped.get_metrics()
        total_rewards.append(episode_reward)
        hit_rates.append(metrics['hit_rate'])
        byte_hit_rates.append(metrics['byte_hit_rate'])
    
    total_actions = sum(action_counts.values())
    action_pcts = {ACTION_NAMES[k]: f"{v/total_actions:.1%}" for k, v in action_counts.items()}
    tier_hits = {k: metrics.get(f'tier_{k}_rate', 0.0) for k in [0, 1]}
    
    return np.mean(total_rewards), np.mean(hit_rates), np.mean(byte_hit_rates), action_pcts, tier_hits


def main():
    parser = argparse.ArgumentParser(description='Train Discrete SAC Agent')
    
    parser.add_argument('--steps', type=int, default=500_000, help='Total training steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=100_000, help='Replay buffer size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--start-steps', type=int, default=10_000, help='Random exploration steps before training')
    parser.add_argument('--alpha', type=float, default=0.2, help='Initial entropy coefficient')
    parser.add_argument('--hidden-dims', type=str, default='256,256', help='Hidden layer dimensions (comma-separated)')
    
    parser.add_argument('--cache-mb', type=float, default=1.0, help='Edge cache size in MB')
    parser.add_argument('--regional-mb', type=float, default=2.0, help='Regional cache size in MB')
    parser.add_argument('--data', type=str, default='data/processed/consistent_trace.csv', help='Path to trace data')
    
    parser.add_argument('--save-path', type=str, default='models/sac_discrete.pt', help='Model save path')
    parser.add_argument('--log-interval', type=int, default=1000, help='Steps between log outputs')
    parser.add_argument('--eval-interval', type=int, default=10000, help='Steps between evaluations')
    
    args = parser.parse_args()
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Steps:         {args.steps:,}")
    print(f"Start Steps:   {args.start_steps:,}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Buffer Size:   {args.buffer_size:,}")
    print(f"Gamma:         {args.gamma}")
    print(f"Hidden Dims:   {hidden_dims}")
    print(f"Edge Cache:    {args.cache_mb} MB")
    print(f"Regional:      {args.regional_mb} MB")
    print(f"Save Path:     {args.save_path}")
    print(f"Device:        {device}")
    print("\n Training has started:")    

    loader = DataLoader(args.data, split_ratio=0.7, mode='train')
    topology = create_simple_hierarchy(
        edge_capacity_mb=args.cache_mb,
        regional_capacity_mb=args.regional_mb
    )
    base_env = CacheEnv(loader, topology=topology)
    env = EpisodeWrapper(base_env, max_steps_per_episode=3000)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DiscreteSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        alpha=args.alpha,
        hidden_dims=hidden_dims,
        automatic_entropy_tuning=True,
        device=device
    )
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    metrics_history = defaultdict(lambda: deque(maxlen=100))
    action_window = deque(maxlen=5000)
    reward_window = deque(maxlen=5000)
    
    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    best_hit_rate = 0.0
    
    for step in range(1, args.steps + 1):
        if step < args.start_steps: #random actions during warmup
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, evaluate=False)
        
        action_window.append(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_window.append(reward)
        
        agent.store_transition(obs, action, reward, next_obs, done)
        
        if step >= args.start_steps:
            metrics = agent.train_step()
            if metrics:
                for k, v in metrics.items():
                    metrics_history[k].append(v)
        
        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        
        if done:
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
        
        if step % args.log_interval == 0:
            avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in metrics_history.items()}
            avg_reward = np.mean(reward_window) if reward_window else 0
            
            current_time = time.time()
            fps = int((step - last_log_step) / (current_time - last_log_time))
            last_log_time = current_time
            last_log_step = step
            
            alpha_val = avg_metrics.get('alpha', agent.alpha)
            
            action_counts = defaultdict(int)
            for a in action_window:
                action_counts[a] += 1
            total_a = len(action_window) or 1
            action_dist = ' | '.join([f"{ACTION_NAMES[i]}:{action_counts[i]/total_a:.1%}" for i in range(action_dim)])
            
            print(f"Step {step:>7}/{args.steps} | "
                  f"α:{alpha_val:.3f} | "
                  f"Q:{avg_metrics.get('mean_q', 0):>6.2f} | "
                  f"Ent:{avg_metrics.get('entropy', 0):>5.2f} | "
                  f"ALoss:{avg_metrics.get('actor_loss', 0):>6.2f} | "
                  f"CLoss:{avg_metrics.get('critic_loss', 0):>6.2f} | "
                  f"R̄:{avg_reward:>6.2f} | "
                  f"FPS:{fps:>3} | {action_dist}")
        
        if step % args.eval_interval == 0:
            print("\n")
            print("Validation:")
            
            eval_reward, eval_hit, eval_byte, action_pcts, tier_hits = evaluate(agent, env)
            
            cache_stats = []
            for node_id in base_env.nodes:
                node = base_env.topology.get_node(node_id)
                cache_stats.append(f"{node_id}:{node.get_occupancy():.0%}")
            cache_info = ', '.join(cache_stats)
            
            print(f"Reward: {eval_reward:.2f} | Hit Rate: {eval_hit:.2%} | Byte Hit: {eval_byte:.2%}")
            print(f"Tier Hits: Edge: {tier_hits.get(0, 0):.1%} | Regional: {tier_hits.get(1, 0):.1%}")
            print(f"Actions: {action_pcts}")
            print(f"Cache Occupancy: {cache_info}")
            

            agent.save(args.save_path)
            if eval_hit > best_hit_rate: #saving best model
                best_hit_rate = eval_hit
                best_path = args.save_path.replace('.pt', '_best.pt')
                agent.save(best_path)
    
    total_time = time.time() - start_time
    print("Training Complete!")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Best Hit Rate: {best_hit_rate:.2%}")
    print(f"Best model saved to: {args.save_path}")
    agent.save(args.save_path)


if __name__ == "__main__":
    main()
