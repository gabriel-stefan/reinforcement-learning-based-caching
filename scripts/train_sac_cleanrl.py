# docs at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
import argparse
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.environments import CacheEnv
from src.environments.core.network_topology import create_simple_hierarchy

@dataclass
class ReplayBufferSamples:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor

class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, handle_timeout_termination=False):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        
        # adjusted shapes for VectorEnv 
        self.obs = np.zeros((buffer_size,) + observation_space.shape, dtype=observation_space.dtype)
        self.next_obs = np.zeros((buffer_size,) + observation_space.shape, dtype=observation_space.dtype)
        self.actions = np.zeros((buffer_size,) + action_space.shape, dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        
        self.pos = 0
        self.full = False
        
    def add(self, obs, next_obs, actions, rewards, dones, infos):
        # obs is (num_envs which is 1, *obs_shape). 
        self.obs[self.pos] = obs[0]
        self.next_obs[self.pos] = next_obs[0]
        self.actions[self.pos] = actions[0]
        self.rewards[self.pos] = rewards[0]
        self.dones[self.pos] = dones[0]
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
            
    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        data = (
            self.obs[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.next_obs[batch_inds],
            self.dones[batch_inds],
        )
        
        return ReplayBufferSamples(*[torch.as_tensor(i).to(self.device) for i in data])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__)[: -len(".py")])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="caching-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    parser.add_argument("--env-id", type=str, default="CacheEnv-v0")
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--buffer-size", type=int, default=int(1e5))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=1e-3)
    parser.add_argument("--update-frequency", type=int, default=4)
    parser.add_argument("--target-network-frequency", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--target-entropy-scale", type=float, default=0.89)
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--model-path", type=str, default=None,
        help="path to save the model (e.g., models/sac.pt)")
    parser.add_argument("--hidden-size", type=int, default=256,
        help="hidden layer size for the neural networks")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    
    args = parser.parse_args()
    return args

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

def make_env(seed, idx, capture_video, run_name):
    def thunk():
        data_path = 'data/processed/consistent_trace.csv'
        loader = DataLoader(data_path, split_ratio=0.7, mode='train')
        loader.reset = lambda: None 
        
        topology = create_simple_hierarchy(
            edge_capacity_mb=1.0,
            regional_capacity_mb=2.0,
            origin_latency_ms=100.0
        )
        env = CacheEnv(data_loader=loader, topology=topology)
        
        env = gym.wrappers.TimeLimit(env, max_episode_steps=3000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        env.action_space.seed(seed)
        return env

    return thunk


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
            
        self.fc1 = layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc3 = layer_init(nn.Linear(hidden_size, action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = Actor(envs, args.hidden_size).to(device)
    qf1 = SoftQNetwork(envs, args.hidden_size).to(device)
    qf2 = SoftQNetwork(envs, args.hidden_size).to(device)
    qf1_target = SoftQNetwork(envs, args.hidden_size).to(device)
    qf2_target = SoftQNetwork(envs, args.hidden_size).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)


        if "episode" in infos:
            for i in range(envs.num_envs):
                if infos["_episode"][i]:
                    # infos["episode"] is a dict of arrays {'r': [...], 'l': [...]}
                    ep_return = infos["episode"]["r"][i]
                    ep_length = infos["episode"]["l"][i]
                    print(f"global_step={global_step}, episodic_return={ep_return}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    
                    if 'hit_rate' in infos:
                         writer.add_scalar("charts/hit_rate", infos['hit_rate'][i], global_step)
                    break
        elif "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                if 'hit_rate' in info:
                     writer.add_scalar("charts/hit_rate", info['hit_rate'], global_step)
                break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
                else:
                    pass
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long().view(-1, 1)).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long().view(-1, 1)).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if args.save_model:
        model_path = args.model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'actor_state_dict': actor.state_dict(),
            'qf1_state_dict': qf1.state_dict(),
            'qf2_state_dict': qf2.state_dict(),
            'qf1_target_state_dict': qf1_target.state_dict(),
            'qf2_target_state_dict': qf2_target.state_dict(),
            'log_alpha': log_alpha if args.autotune else None,
            'args': vars(args),
            'hidden_size': args.hidden_size
        }, args.model_path)
        print(f"Model saved to {args.model_path}")

    envs.close()
    writer.close()
