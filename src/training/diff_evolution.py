# backpropagation.py
# Defines the back propagation algorithm for learning a neural network.

import math
import numpy as np
import random as rd
import src.util.activations as af
from src.network import Network
from src.data import *
from functools import reduce

class DiffEvolution:
    # DE is a population based optimizer that perturbs vectors using scaled differences of randomly generated
    # individual vectors

    def __init__(self, network: Network, mutationF: float, recombinationC: float, popsize=30):
        self.network = network  # Minimize the objective function by optimizing the values of the network weights
        self.mutationF = mutationF  # Mutation rate
        self.recombinationC = recombinationC  # Recombination rate
        self.population_size = popsize
        self.population = self.initialize_population()  # Initialize Population
        self.run_nn_on_pop_weights()
        self.test_pop = self.population  # used so we can compare old weights with new weights
        self.individual_feature_size = self.network.get_num_weights()  # Individuals features

        self.final_performance = 0
        self.run()

    def run_nn_on_pop_weights(self):
        fitness = []
        cnt = 0
        for individual in self.population:
            self.network.weights = individual
            fitness.append(self.get_fitness())
            cnt += 1
        print(fitness)

    def initialize_population(self):
        bounds = -2000, 2000
        population = []
        print("Individual Feature Size" + str(self.individual_feature_size))
        for individual in range(self.population_size):
            individual = np.random.uniform(low=bounds[0], high=bounds[1],
                                            size=self.individual_feature_size)  # Create Individual
            population.append(individual)  # Add individual to population
        return population


    def mutation(self, loc):
        """Mutation is used to allow us to explore our space to find a good solution. To do this we select three
        vectors at random from the current population."""
        # Get random index's of the population were going changing
        mutation_idx = []
        for _ in range(3):
            value = rd.sample(range(0, len(self.population)-1), 3)
            while value == loc:
                value = rd.sample(range(0, len(self.population)-1), 3)
            mutation_idx.append(value)

        # Mutate them
        print("Mutation Idx: " + str(mutation_idx))
        print("Mutation Idx Type: " + type(mutation_idx))
        first_chosen_one = np.asarray(self.population[mutation_idx[0]])
        second_chosen_one = np.asarray(self.population[mutation_idx[1]])
        third_chosen_one = np.asarray(self.population[mutation_idx[2]])
        mutant = first_chosen_one + self.mutationF * (second_chosen_one - third_chosen_one)
        return mutant

    def recombination(self, loc, mutant):
        """Do we perform crossover"""
        for i in range(self.individual_feature_size):
            we_do_replace = random.uniform(0, 1) < self.recombinationC
            if we_do_replace:
                self.test_pop[loc][i] = mutant[i]

    def run(self):
        iteration = 1000
        for i in range(iteration):  # For i amt of iterations
            for loc in range(self.population - 1):  # For each location
                mutant = self.mutation(loc)  # Mutate
                self.recombination(loc, mutant)  # Perform crossover
                # update weights of network
                old_performance = self.get_fitness()
                self.network.weights = self.test_pop  # Test with new weights
                new_performance = self.get_fitness()
                self.network.weights = self.population  # Change back just in case

                if old_performance < new_performance:  # if performance better we replace the population with the mutant
                    self.network.weights = self.test_pop  # These weights were better
                    self.population = self.test_pop  # Lets update population to reflect this
                    self.final_performance = new_performance
                self.final_performance = old_performance
        return self.final_performance

    def encode(self):
        weights = []
        i = 0
        for shape in self.network.get_weight_shapes():
            size = reduce((lambda x, y: x * y), shape)
            weights.append(np.reshape(self.state[i:i + size], shape))
            i += size
        return weights

    # Returns the fitness of the individuals current state. The fitness is evaluated on the training set for the network.
    # The fitness is accuracy if a classification problem, and the inverse error for regression.
    def get_fitness(self):
        old_weights = self.network.weights #TODO update to work with your code
        self.network.weights = self.encode()
        if self.network.is_regression():
            fitness = 1 / self.network.get_error(self.network.training_data)
        else:
            fitness = self.network.get_accuracy(self.network.training_data)
        self.network.weights = old_weights
        return fitness

    def weights_to_vector(self):
        pass


dataset = get_machine_data()
trainset, testset = dataset.partition(.80)
#print(trainset)
# test with number of features as size of input layer, guessing for hidden, and 1 output node size as regression is used
n = Network(trainset, testset, [6, 3, 3, 3, 1])
x = DiffEvolution(n, .1, .9, 39)

