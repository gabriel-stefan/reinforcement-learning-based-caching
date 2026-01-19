# Deep Reinforcement Learning for CDN Content Placement

<p align="left">
  <strong>Using Deep Reinforcement Learning to Optimize Content Placement in Hierarchical Content Delivery Networks</strong>
</p>

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Environment Design](#environment-design)
- [Baselines](#baselines)
- [Agent Implementations](#agent-implementations)
- [Experiments](#experiments)
- [Conclusions](#conclusions)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Project Overview

I built this project to explore whether Deep Reinforcement Learning can make intelligent content placement decisions in Content Delivery Networks. While traditional caching research focuses mainly on eviction policies, I explored the idea that content placement is equally important and well suited for Reinforcement Learning.

The environment is designed to be fully configurable, allowing users to define custom network topologies with any number of tiers, capacities, and latency/bandwidth characteristics.

The default configuration used in this project is a three-tier hierarchical network:
- **Edge (Tier 0)**: Low latency (1ms), high bandwidth (1000 Mbps), limited storage (1 MB)
- **Regional (Tier 1)**: Intermediate latency (10ms), medium bandwidth (500 Mbps), more storage (2 MB)
- **Origin (Tier 2)**: High latency (100ms), lower bandwidth (100 Mbps), unlimited storage

The Reinforcement Learning agent learns to decide, for each incoming request, whether it should cache the item at the Edge, at the Regional cache, or skip caching it entirely.

## Data 

I use the Wikipedia request traces from the paper, which are real HTTP requests from Wikipedia servers:

> *Wikipedia Workload Analysis for Decentralized Hosting*  
> Guido Urdaneta, Guillaume Pierre, and Maarten van Steen  
> [Paper Link](http://www.globule.org/publi/WWADH_comnet2009.html)

The data is split 70/30 for training/testing to evaluate generalization.

## Environment Design

### Custom Gymnasium Environment

I built a custom Gymnasium environment that simulates the hierarchical Content Delivery Network. 

#### Observation Space 
For each cache tier:
- **Occupancy**: How full is this cache (0-1)
- **Number of items**: Normalized item count
- **Average item size**: Mean size of cached content
- **Average frequency**: Mean access frequency of cached items
- **Average recency**: Exponential decay based on last access time
- **Cache pressure**: Ratio of incoming item size to free space

Global features:
- **Incoming content size** (normalized with log transform)
- **Frequency hint**: How many times we've seen this URL before
- **Current hit rate**: Running hit rate for the episode
- **Tier hit ratios**: What fraction of hits came from each tier

#### Action Space
Discrete action space with 3 actions:

- Cache at Edge
- Cache at Regional  
- Skip (don't cache anywhere)


#### Reward Function: Latency-Based

The idea was to make the rewards based on actual latency savings.

```python
# Calculate latency for origin fetch
origin_latency = topology.calculate_origin_fetch_time(size)

# Calculate actual latency based on where we served from
actual_latency = lookup.total_latency_ms

# Reward = normalized latency savings × scale factor
savings_ratio = (origin_latency - actual_latency) / origin_latency
reward = savings_ratio * 10.0
```

I chose rewards based on latency because they are grounded in real performance metrics that directly reflect user experience. Latency-based rewards remain effective and consistent across different network configurations.


## Baselines

To evaluate whether the Reinforcement Learning approach adds value, I implemented several heuristic placement policies:

- **EdgeFirst**  
  Always cache content at the Edge tier.

- **SizeSplit (Median)**  
  Cache smaller items at the Edge and larger items at the Regional tier, using the median content size as the threshold.

- **PercentileSplit (P90)**  
  Similar to SizeSplit, but uses the 90th percentile of content size as the threshold.

- **Probabilistic**  
  Cache content at the Edge with probability *p = 0.1*; otherwise, skip caching entirely.


### Baseline Results (1 Million Requests)

| Policy | Hit Rate | Total Reward |
|--------|----------|--------------|
| **SizeSplit (Median)** | **52.06%** | **4,472,646** |
| PercentileSplit (P90) | 46.62% | 4,275,183 |
| EdgeFirst | 43.10% | 4,017,288 |
| Probabilistic (10%) | 42.67% | 3,721,054 |


## Agent Implementations

#### 1. DQN  

Implemented using Stable-Baselines3.

#### 2. PPO  

Implemented using Stable-Baselines3.

#### 3. Discrete SAC  

Implementation: both a custom version and an adaptation based on CleanRL.

#### 3.1 Custom implementation   

Standard SAC is designed for continuous action spaces, so I adapted it for discrete actions following Christodoulou (2019): https://arxiv.org/abs/1910.07207

The Actor network outputs a categorical probability distribution over the three actions (Edge, Regional, Skip) using a softmax output layer, while the Critic utilizes twin Q-networks to estimate Q-values for all possible actions simultaneously. During training, the critic minimizes the Bellman error using a soft state-value calculation that incorporates the policy's entropy, and the actor is updated to minimize the KL divergence between the policy and the soft Q-function. To ensure robust exploration without manual tuning, I implemented automatic entropy adjustment, where the temperature parameter α is dynamically optimized to maintain a target entropy level throughout training.

#### 3.2 CleanRL implementation 
I adapted the standard continuous SAC implementation from the CleanRL repository for our discrete action space. The actor now outputs probabilities for each action, while the critic estimates Q-values for all actions simultaneously. 

All training, evaluation, and analysis scripts were implemented by me.

> Note: The best performing Discrete SAC agent was achieved using the CleanRL adaptation, which proved more stable and effective than the custom implementation for this environment.

## Experiments

I ran extensive experiments, tuning parameters and training all agents locally. In total, this amounted to roughly 200 hours of training across all agents. The results below represent the best-performing models, compared against the heuristic baselines.

| Rank | Method | Type | Hit Rate |
|:----:|--------|------|----------|
| 1 | SizeSplit (Median) | Heuristic | 52.06% |
| 2 | PPO (Reduced Clip) | RL | 51.97% |
| 3 | Discrete SAC (CleanRL) | RL | 51.79% |
| 4 | DQN (Best config) | RL | 51.70% |
| 5 | Discrete SAC (Custom) | RL | 49.63% |
| 6 | PercentileSplit (P90) | Heuristic | 46.62% |
| 7 | EdgeFirst | Heuristic | 43.10% |
| 8 | Probabilistic (10%) | Heuristic | 42.67% |

### Detailed Best Agent Performance

Below are the detailed metrics for the best performing configuration of each agent type.

#### 1. PPO
- **Hit Rate**: 51.97%
- **Byte Hit Rate**: 51.81%
- **Average Latency**: 55.41 ms
- **Action Distribution**: 
  - Edge: 51.4%
  - Regional: 30.6%
  - Skip: 18.0%

#### 2. Discrete SAC 
- **Hit Rate**: 51.79%
- **Byte Hit Rate**: 52.06%
- **Average Latency**: 56.13 ms
- **Action Distribution**: 
  - Edge: 43.5%
  - Regional: 8.1%
  - Skip: 48.4%

#### 3. DQN 
- **Hit Rate**: 51.70%
- **Latency**: 55.25 ms
- **Action Distribution**: 
  - Edge: 4%
  - Regional: 96%
  - Skip: 0%


## Conclusions

This project demonstrated that Deep Reinforcement Learning can effectively optimize content placement in hierarchical CDNs. Reinforcement Learning agents like PPO and Discrete SAC achieved comparable hit rates with the best heuristic baseline while learning complex placement strategies. The results suggest that Reinforcement Learning is a viable approach for CDN caching, especially in scenarios where traffic patterns are complex and dynamic.

### Future Directions

- Test on non-stationary data where Reinforcement Learning adaptation could provide advantage
- Explore larger cache hierarchies with more than 2 tiers
- Add network-level features 
- Experiment with multi-agent Reinforcement Leaning architectures 

## Installation

### Prerequisites
- Python 3.10+
- Conda or pip

### Setup
```bash
# Clone repository
git clone https://github.com/gabriel-stefan/reinforcement-learning-based-caching
cd caching_deepRL

# Create conda environment
conda env create -f environment.yml
conda activate caching_deeprl

# Or with pip
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train Discrete SAC (CleanRL)
python scripts/train_sac_cleanrl.py --total-timesteps 500000 --model-path models/sac.pt

# Train DQN (Stable Baselines3)
python scripts/dqn_train.py --steps 500000 --save-path models/dqn

# Train PPO (Stable Baselines3)
python scripts/ppo_train.py --steps 500000 --save-path models/ppo
```

### Evaluation

```bash
# Evaluate SAC model
python scripts/eval_sac_cleanrl.py --model-path models/sac.pt --steps 100000

# Run baseline comparison
python scripts/benchmark_cdn_baselines.py --steps 1000000
```

Note: Each script has its own command-line flags (e.g., timesteps, learning rate, buffer sizes, seeds, model paths, etc.).  

Run any script with `-h` or `--help` to see the available options.

## Citation

If you use this work, please cite:

```bibtex
@misc{caching_deeprl,
  author       = {Gabriel Stefan},
  title        = {Deep Reinforcement Learning for CDN Content Placement},
  year         = {2026},
  howpublished = {\url{https://github.com/gabriel-stefan/reinforcement-learning-based-caching}},
  note         = {GitHub repository}
}
```

### Data Source
```bibtex
@Article{urdaneta2009wikipedia,
  author = 	 {Urdaneta, Guido and Pierre, Guillaume and van Steen, Maarten},
  title = 	 {Wikipedia Workload Analysis for Decentralized Hosting},
  volume =       {53},
  number =       {11},
  pages =        {1830-1845},
  month =        {July},
  year = 	 {2009},
  journal = 	 {Elsevier Computer Networks},
  note = 	 {\url{http://www.globule.org/publi/WWADH_comnet2009.html}}
}
```

## License

MIT License