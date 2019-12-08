# backpropagation.py
# Defines the back propagation algorithm for learning a neural network.

import math
import numpy as np
import random as rd
import src.util.activations as af
from src.network import Network
from src.data import *

class DiffEvolution:
    # DE is a population based optimizer that perturbs vectors using scaled differences of two randomly generated population vectors

    def __init__(self, network: Network, bounds: list, mutationF: float, recombinationC: float):
        self.network = network  # Minimize the objective function by optimizing the values of the network weights
        self.bounds = bounds  # Boundary to limit search space, speeding up algorithm
        self.mutationF = mutationF  # Mutation rate
        self.recombinationC = recombinationC  # Recombination rate
        self.weight_size = self.network.get_num_weights()  # Size of desired population
        self.population = np.random.uniform(low=bounds[0], high=bounds[1], size=self.weight_size)  # Initialize Population
        self.test_pop = self.population  # used so we can compare old weights with new weights
        self.final_performance = 0
        self.run()

    def mutation(self, loc):
        """Mutation is used to allow us to explore our space to find a good solution. To do this we select three
        vectors at random from the current population."""
        # Get random index's were changing
        mutation_idx = []
        for _ in range(3):
            value = rd.sample(range(0, len(self.population)-1), 3)
            while value == loc:
                value = rd.sample(range(0, len(self.population)-1), 3)
            mutation_idx.append(value)

        # Mutate them
        mutant = mutation_idx[0] + self.mutationF * (mutation_idx[1] - mutation_idx[2])
        return mutant

    def recombination(self, loc, mutant):
        """Do we perform crossover"""
        we_do_replace = random.uniform(0, 1) < self.recombinationC
        if we_do_replace:
            self.test_pop[loc] = mutant

    def run(self):
        iteration = 1000
        for i in range(iteration):  # For i amt of iterations
            for loc in range(self.population - 1):  # For each location
                mutant = self.mutation(loc)  # Mutate
                self.recombination(loc, mutant)  # Perform crossover
                #TODO run neural network with new weights and see if it performs better than old.
                # If yes than keep new weights, else discard weights
                # update weights of network
                old_performance = self.network.run()
                self.network.weights = self.test_pop  # Test with new weights
                new_performance = self.network.run()
                self.network.weights = self.population  # Change back just in case

                if old_performance < new_performance:  # if performance better we replace the population with the mutant
                    self.network.weights = self.test_pop  # These weights were better
                    self.population = self.test_pop  # Lets update population to reflect this
                    self.final_performance = new_performance
                self.final_performance = old_performance
        return self.final_performance

    def weights_to_vector(self):
        pass
def get_bounds(network):
    """Returns the lower and upper bounds of the weight parameters that are going to be passed into differential
    evolution. This will reduce our search space... in a good way?"""
    max_value = 0
    min_value = 10000
    for weight_layer in network.weights:
        for all_weights in weight_layer:
            for weight in all_weights:
                if weight > max_value:
                    max_value = weight
                if weight < min_value:
                    min_value = weight
    boundary = [min_value, max_value]
    return boundary


dataset = get_machine_data()
trainset, testset = dataset.partition(.80)
#print(trainset)
# test with number of features as size of input layer, guessing for hidden, and 1 output node size as regression is used
n = Network(trainset, testset, [6, 3, 3, 3, 1])
bounds = get_bounds(n)
x = DiffEvolution(n, bounds, .1, .9)

