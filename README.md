# Caching with Deep Reinforcement Learning

## Project Overview
This project explores the application of Deep Reinforcement Learning to optimize content caching in a hierarchical Content Delivery Network.

Instead of trying to learn a eviction policy from scratch, the problem is split into two parts:
1.  Eviction: Used a fixed, high-performance heuristic for this trace (LFU) to handle eviction when the cache is full.
2.  Placement: Trained various Deep Reinforcement Learning Agents to make Placement Decisions depinding on content.

## Why LFU for Eviction?
Conducted benchmarking of standard eviction policies on the specific Wikipedia trace. The results demonstrated that Least Frequently Used outperforms other strategies, confirming that frequency is the dominant signal for this Zipfian data.

### Benchmark Results (1 Million Steps)
*Simulated on a Unified Cache (Edge + Regional capacity) to test pure eviction performance.*

| Policy | Hit Rate | Byte Hit Rate | Time |
| :--- | :--- | :--- | :--- |
| **LFU** | **52.18%** | **51.10%** | 102.0s | 
| LRU | 40.30% | 39.15% | 92.5s | 
| FIFO | 34.91% | 33.81% | 95.4s | 
| Random | 34.89% | 33.79% | 47.7s | 


## Data
Uses the Wikipedia traces from *Wikipedia Workload Analysis for Decentralized Hosting* by Guido Urdaneta, Guillaume Pierre, Maarten van Steen.
