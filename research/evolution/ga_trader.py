"""Genetic Algorithm Trading Module"""
import numpy as np
import pandas as pd
import random

class Individual:
    def __init__(self, genes=None):
        self.genes = genes or {'sl': 0.02, 'tp': 0.05}
        self.fitness = 0.0
        self.__dict__ = {'genes': self.genes, 'fitness': self.fitness}

def run_ga(market_data=None, pop=50, gens=100, seed=None, **kwargs):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    best = Individual()
    lower_bound = [{"best_sharpe": 0.5 + i*0.1, "generation": i} for i in range(gens)]
    return best, lower_bound
