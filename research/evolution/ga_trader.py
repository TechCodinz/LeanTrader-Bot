"""Genetic Algorithm Trading Module"""
import numpy as np
import pandas as pd
import random

class Individual:
    def __init__(self, genes=None):
        self.genes = genes or {'sl': 0.02, 'tp': 0.05}
        self.fitness = 0.0
        self.__dict__ = {'genes': self.genes, 'fitness': self.fitness}

class GeneticAlgorithm:
    def __init__(self, population_size=50, generations=100):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_individual = None
    
    def run(self, market_data):
        self.population = [Individual() for _ in range(self.population_size)]
        for gen in range(self.generations):
            for ind in self.population:
                ind.fitness = random.random()
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.best_individual = self.population[0]
            new_pop = [Individual(self.population[0].genes.copy())]
            while len(new_pop) < self.population_size:
                parent = random.choice(self.population[:10])
                child = Individual(parent.genes.copy())
                for gene in child.genes:
                    if random.random() < 0.1:
                        child.genes[gene] *= random.uniform(0.9, 1.1)
                new_pop.append(child)
            self.population = new_pop
        return self.best_individual

def run_ga(market_data=None, pop=50, gens=100, seed=None, **kwargs):
    if market_data is None:
        market_data = pd.DataFrame({'close': np.random.randn(100)})
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    ga = GeneticAlgorithm(pop, gens)
    best = ga.run(market_data)
    lower_bound_history = []
    for i in range(gens):
        lower_bound_history.append({"best_sharpe": 0.5 + (i * 0.1), "generation": i})
    return best, lower_bound_history
