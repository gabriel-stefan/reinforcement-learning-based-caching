import numpy as np
import random

class edge_first:
    """always attempts to cache at edge (action 0)."""
    def __init__(self): pass
    def predict(self, obs, state=None, deterministic=True):
        return 0, state

class size_split:
    """caches small items at edge, large at regional. default threshold is median (2463b)."""
    def __init__(self, threshold=2463, max_size_norm=5000000):
        self.threshold = threshold
        self.norm_threshold = np.log1p(threshold) / np.log1p(max_size_norm)
        
    def predict(self, obs, state=None, deterministic=True):
        # obs structure: [..., size, freq, ...]
        # size is at index -5
        norm_size = obs[-5]
        if norm_size < self.norm_threshold:
            return 0, state # edge
        return 1, state # regional

class probabilistic:
    """caches at edge with probability p, else skips. default p=0.1."""
    def __init__(self, p=0.1):
        self.p = p
        
    def predict(self, obs, state=None, deterministic=True):
        if random.random() < self.p:
            return 0, state # edge
        return 2, state # skip

class percentile_split:
    """caches items < p90 size at edge. default threshold is p90 (4299b)."""
    def __init__(self, threshold=4299, max_size_norm=5000000):
        self.threshold = threshold
        self.norm_threshold = np.log1p(threshold) / np.log1p(max_size_norm)
        
    def predict(self, obs, state=None, deterministic=True):
        norm_size = obs[-5]
        if norm_size < self.norm_threshold:
            return 0, state # edge
        return 1, state # regional
