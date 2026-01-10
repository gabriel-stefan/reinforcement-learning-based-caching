# used for reference: https://arxiv.org/abs/1910.07207

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional


class ReplayBuffer:
    
    def __init__(self, capacity: int, state_dim: int = None):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self._state_dim = state_dim
        self._initialized = False
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    def _initialize(self, state):
        state_shape = np.asarray(state).shape
        self._state_dim = state_shape[0] if len(state_shape) > 0 else 1
        
        self.states = np.zeros((self.capacity, self._state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self._state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._initialized = True

    def add(self, state, action, reward, next_state, done):
        if not self._initialized:
            self._initialize(state)
            
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size

class Actor(nn.Module):
    
    #outputting a probability distribution over actions using softmax

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super(Actor, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        return probs


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super(Critic, self).__init__()
        
        # first Q-network
        layers1 = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers1.append(nn.Linear(prev_dim, dim))
            layers1.append(nn.ReLU())
            prev_dim = dim
        layers1.append(nn.Linear(prev_dim, action_dim))
        self.q1 = nn.Sequential(*layers1)
        
        # second Q-network 
        layers2 = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers2.append(nn.Linear(prev_dim, dim))
            layers2.append(nn.ReLU())
            prev_dim = dim
        layers2.append(nn.Linear(prev_dim, action_dim))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state) #returns Q-values for all actions from both critics

class DiscreteSACAgent:
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_interval: int = 1,
        hidden_dims: list = [256, 256],
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = torch.device(device)


        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device) #networks
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        if self.automatic_entropy_tuning:
            self.target_entropy = -0.98 * np.log(1.0 / action_dim) #to discourage exploration
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        
        self.memory = ReplayBuffer(buffer_size)
        self.steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.actor(state_t)
            
            if evaluate:
                action = torch.argmax(probs, dim=1) #greedily choose action
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
            return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.memory) < self.batch_size:
            return None

        self.steps += 1
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)


        # computing target Q-value using target networks
        with torch.no_grad():
            # next state action probabilities from current policy
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8) 
            
            # geting the Q-values from target critics
            next_q1, next_q2 = self.critic_target(next_states)
            next_q = torch.min(next_q1, next_q2)  # Take min to reduce overestimation
            
            #calculate soft state-value
            next_v = torch.sum(next_probs * (next_q - self.alpha * next_log_probs), dim=1, keepdim=True)
            
            target_q = rewards + (1 - dones) * self.gamma * next_v

        current_q1, current_q2 = self.critic(states)
        current_q1 = current_q1.gather(1, actions)
        current_q2 = current_q2.gather(1, actions)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()


        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        
        with torch.no_grad():
            q1, q2 = self.critic(states)
            min_q = torch.min(q1, q2)
        
       
        actor_loss = torch.sum(probs * (self.alpha * log_probs - min_q), dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()


        alpha_loss = torch.tensor(0.0)
        entropy = torch.tensor(0.0)
        
        if self.automatic_entropy_tuning:
            with torch.no_grad():
                entropy = -torch.sum(probs * log_probs, dim=1).mean()

            alpha_loss = self.log_alpha * (entropy - self.target_entropy)
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()

        if self.steps % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item() if self.automatic_entropy_tuning else 0.0,
            'alpha': self.alpha,
            'entropy': entropy.item() if self.automatic_entropy_tuning else 0.0,
            'mean_q': current_q1.mean().item()
        }

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'steps': self.steps
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        
        if 'critic_target' in checkpoint:
            self.critic_target.load_state_dict(checkpoint['critic_target'])
        else:
            self.critic_target.load_state_dict(checkpoint['critic'])
            
        if self.automatic_entropy_tuning and checkpoint.get('log_alpha') is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
            self.alpha = self.log_alpha.exp().item()
            
        if 'steps' in checkpoint:
            self.steps = checkpoint['steps']  
