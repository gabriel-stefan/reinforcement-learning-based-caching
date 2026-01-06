import os
from typing import Optional, Dict, Any
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class PlacementDQN:
    def __init__(self, env, model_path: str = "models/dqn", 
                 learning_rate: float = 1e-4, 
                 buffer_size: int = 100000,
                 learning_starts: int = 1000,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 exploration_fraction: float = 0.1,
                 exploration_final_eps: float = 0.05,
                 verbose: int = 1):
        
        self.env = env
        self.model_path = model_path
        self.verbose = verbose
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            verbose=verbose,
            tensorboard_log="./tensorboard_logs/"
        )
        
    def train(self, total_timesteps: int = 100000, log_interval: int = 100):
        print(f"Training for {total_timesteps} steps...")
        self.model.learn(
            total_timesteps=total_timesteps, 
            log_interval=log_interval,
            callback=None,
            progress_bar=True
        )
        print("Training complete.")
        
    def save(self, path: Optional[str] = None):
        save_path = path or self.model_path
        self.model.save(save_path)
        print(f"Saved to {save_path}")
        
    @classmethod
    def load(cls, path: str, env=None, **kwargs):
        if not os.path.exists(path) and not os.path.exists(path + ".zip"):
            raise FileNotFoundError(f"No model found at {path}")
            
        print(f"Loading model from {path}...")
        instance = cls(env, model_path=path, **kwargs)
        instance.model = DQN.load(path, env=env)
        return instance
        
    def predict(self, obs, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)
