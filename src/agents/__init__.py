from .baselines import (
    EvictionPolicy,
    RandomPolicy,
    LRUPolicy,
    LFUPolicy,
    FIFOPolicy,
    SizePolicy,
    POLICIES,
    get_policy
)


__all__ = [
    'EvictionPolicy',
    'RandomPolicy',
    'LRUPolicy',
    'LFUPolicy',
    'FIFOPolicy',
    'SizePolicy',
    'POLICIES',
    'get_policy',
]
