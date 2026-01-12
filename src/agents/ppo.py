import os
from typing import Optional, Dict, Any
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class PlacementPPO:
    def __init__(self, env, model_path: str = "models/ppo", 
                 learning_rate: float = 3e-4, 
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 1):
        
        self.env = env
        self.model_path = model_path
        self.verbose = verbose
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,

            policy_kwargs=policy_kwargs,
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
        instance.model = PPO.load(path, env=env)
        return instance
        
    def predict(self, obs, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)
