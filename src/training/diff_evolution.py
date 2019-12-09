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

    # Creates an instance of the differential evolution (DE) algorithm. The DE trains the given network according to the
    # given parameters.
    # * network: the network object to train. (fitness function / objective function)
    # * pop_size: the size of the population; the number of particles.
    # * mutationF: the coefficient influencing how much to bias amount of mutation occurring.
    # * recombinationC: the coefficient influencing how much cross over occurs between the individual and mutant.
    def __init__(self, network: Network, mutationF: float, recombinationC: float, popsize=30):
        self.network = network  # Minimize the objective function by optimizing the values of the network weights
        self.mutationF = mutationF  # Mutation rate
        self.recombinationC = recombinationC  # Recombination rate
        self.population_size = popsize
        self.individual_feature_size = self.network.get_num_weights()  # Individuals features
        self.population = self.initialize_population()  # Initialize Population
        self.run_nn_on_pop_weights()
        self.test_pop = self.population  # used so we can compare old weights with new weights
        self.run()

    def run_nn_on_pop_weights(self):
        """Uncomment for testing"""
        # fitness = []
        # cnt = 0
        # for individual in self.population:
        #     print(individual)
        #     print(self.get_fitness(individual))
        #     fitness.append(self.get_fitness(individual))
        #     cnt += 1
        # print(fitness)
        pass

    def initialize_population(self):
        """Initialize Population by randomly selecting feature values for each individual within the population."""
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
        individuals at random from the current population. We then perform mutation and creating our mutant"""
        # Get random index's of the population were going changing
        # mutation_idx = []
        # for _ in range(1):
        mutation_idx = rd.sample(range(0, len(self.population)-1), 3)
        #     while value == loc:
        #         value = rd.sample(range(0, len(self.population)-1), 3)
        #     mutation_idx.append(value)
        # Mutate them
        first_chosen_one = np.asarray(self.population[mutation_idx[0]])
        second_chosen_one = np.asarray(self.population[mutation_idx[1]])
        third_chosen_one = np.asarray(self.population[mutation_idx[2]])
        mutant = first_chosen_one + self.mutationF * (second_chosen_one - third_chosen_one)
        return mutant

    def recombination(self, loc, mutant):
        """Perform crossover on each of our features within an individual"""
        for i in range(self.individual_feature_size):
            we_do_replace = random.uniform(0, 1) > self.recombinationC
            if we_do_replace:
                #print("We replaced" + str(i))
                self.test_pop[loc][i] = mutant[i]


    def run(self):
        """Runs the differential evolution optimization on each individual and finds the most fit individual within
        the space."""
        iteration = 20
        best_performance = 0
        best_individual = []
        for i in range(iteration):  # For i amt of iterations
            for loc in range(len(self.population) - 1):  # For each individual
                mutant = self.mutation(loc)  # Mutate
                #print("Mutant: " + str(mutant))
                individual = self.population[loc]
                #print("Individual: " + str(individual))
                self.recombination(loc, mutant)  # Perform crossover
                #print("Test Population: " + str(self.test_pop[loc]))

                old_performance = self.get_fitness(individual)
                #print("Old Performace: " + str(old_performance))
                test_pop_individual = self.test_pop[loc]  # Test with new weights
                new_performance = self.get_fitness(test_pop_individual)
                #print("New Performace: " + str(new_performance))
                #print("-----------------")

                if new_performance > best_performance:
                    best_performance = new_performance
                    best_individual = self.test_pop[loc]

                if old_performance < new_performance:  # if performance better we replace the population with the mutant
                    self.network.weights = self.test_pop[loc]  # These weights were better
                    self.population = self.test_pop  # Lets update population to reflect this

        #print(best_performance)
        #print(best_individual)
        self.network.weights = best_individual

    def encode(self, individual):
        """Encodes a vector into matricies that represent the Feed Forward Neural Network"""
        weights = []
        i = 0
        for shape in self.network.get_weight_shapes():
            size = reduce((lambda x, y: x * y), shape)
            weights.append(np.reshape(individual[i:i + size], shape))
            i += size
        return weights

    # Returns the fitness of the individuals current state. The fitness is evaluated on the training set for the network.
    # The fitness is accuracy if a classification problem, and the inverse error for regression.
    def get_fitness(self, individual):
        """Determine the fitness of an individual"""
        old_weights = self.network.weights
        self.network.weights = self.encode(individual)
        if self.network.is_regression():
            fitness = 1 / self.network.get_error(self.network.training_data)
        else:
            fitness = self.network.get_accuracy(self.network.training_data)
        self.network.weights = old_weights
        return fitness


# dataset = get_machine_data()
# trainset, testset = dataset.partition(.80)
# #print(trainset)
# # test with number of features as size of input layer, guessing for hidden, and 1 output node size as regression is used
# n = Network(trainset, testset, [6, 3, 3, 3, 1])
# x = DiffEvolution(n, .1, .9, 40)


