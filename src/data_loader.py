import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, csv_path, split_ratio=0.7, mode='train'):
        print(f"Loading data from {csv_path} for [{mode}]")
        self.full_df = pd.read_csv(csv_path)
        
        split_index = int(len(self.full_df) * split_ratio)
        
        if mode == 'train':
            self.data = self.full_df.iloc[:split_index].reset_index(drop=True)
        elif mode == 'test':
            self.data = self.full_df.iloc[split_index:].reset_index(drop=True)
        else:
            raise ValueError("Mode must be 'train' or 'test'")
            
        self.total_len = len(self.data)
        self.pointer = 0
        print(f"Data Loaded: {self.total_len} requests.")

    def reset(self):
        self.pointer = 0
    
    def _get_simulated_latency(self, url: str) -> float:
        h = int(hash(url))
        return 10 + (abs(h) % 491)

    def get_next(self):
        if self.pointer >= self.total_len:
            return None
        
        row = self.data.iloc[self.pointer]
        self.pointer += 1
        
        url = row['url']
        size = int(row['size'])
        timestamp = row['timestamp']
        
        cost = self._get_simulated_latency(url)
        
        return url, size, timestamp, cost
        
    def __len__(self):
        return self.total_len