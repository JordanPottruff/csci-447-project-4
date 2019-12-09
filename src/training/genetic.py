# genetic.py
# Implements genetic algorithm with neural networks

import random
import itertools
from functools import reduce
import math
import numpy
import numpy as np
from src.network import Network

# Search space for the neural network weights, tunable
LOWER_BOUND = -100000
UPPER_BOUND = 100000

# The amount of fitness increase used in the average metric to continue the learning process
CONVERGENCE_THRESHOLD = .0001


# Creates an individual in a population. Each weight matrix of the neural network is represented by an individual
# denoted by a vector. An individual takes in a network, used to construct the length of the vector and calculate
# the fitness, the vector (initally set to random), and each vector is predetermined to mutate. The individual class
# holds instance of the network, fitness, and the vector.
class Individual:
    def __init__(self, network: Network, vector, mutation_prob=0.0, creep_variance=1.0):
        self.network = network
        self.vector = vector
        for gene in range(len(self.vector)):
            if flip(mutation_prob):
                self.vector[gene] += np.random.normal(0, creep_variance)
        self.fitness = self.calc_fitness()

    # In classification, the fitness is the accuracy of the network. Therefore, we would want to maximize the accuracy
    # of the population. Inversely, in a regression data set, we would want to reduce and minimize the error. The error
    # is calculated by the inverse accuracy. Here, both metric of fitness is desired to be maximized.
    def calc_fitness(self):
        weight_matrix = self.vector_to_matrix(self.vector)
        self.network.weights = weight_matrix
        if self.network.is_regression():
            fitness = 1 / self.network.get_error(self.network.training_data)
        else:
            fitness = self.network.get_accuracy(self.network.training_data)
        return fitness

    # Encodes a vector into the proper np.array shape of the network. Used for testing and finding the accuracy of the
    # weights.
    def vector_to_matrix(self, vector):
        weights = []
        i = 0
        for shape in self.network.get_weight_shapes():
            size = reduce((lambda x, y: x * y), shape)
            weights.append(np.reshape(vector[i:i+size], shape))
            i += size
        return weights


# Coin simulation based on a probability.
def flip(prob):
    return True if random.random() < prob else False


# Implementation class for the genetic algorithm. The class trains the neural network by simulating many weights,
# converges to the better fitness individual by crossover, mutation, selection, and replacement. The following genetic
# algorithm implements binomial crossover, creep mutation from a normal distribution, tournament-based selection, and
# generational replacement as the parameters. The genetic algorithm must be initiated by executing the train method.
class Genetic:
    def __init__(self, network: Network, population_size: int, crossover_prob: float,
                 creep_variance: float, mutation_prob: float, tournament_size: int, convergence_size: int):
        self.network = network
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.creep_variance = creep_variance
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.convergence_size = convergence_size
        self.population = [Individual(network, np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND,
                                                                 size=network.get_num_weights()))
                           for i in range(population_size)]

    # Returns the highest fitness individual from the population.
    def get_highest_fitness(self):
        max_fitness = 0
        max_fit_individual = self.population[0]
        for ind in self.population:
            if ind.fitness > max_fitness:
                max_fitness = ind.fitness
                max_fit_individual = ind
        return max_fit_individual.fitness

    # Tournament selection. Random k individuals are selected for a tournament with replacement. The individual with
    # the highest fitness will be selected as parents and subsequently placed into the mating pool for future
    # generations and operations.
    def parent_selection(self):
        parent_pairs = []
        # Since we implemented generational replacement, the number of offspring must be the same number as the
        # population size as the entire population will be replaced.
        num_offsprings = int(self.population_size)
        for num in range(math.ceil(num_offsprings / 2)):
            winners = []
            for i in range(2):
                # Simulate a tournament of size k. Selects the fittest individual
                tournament = [random.choice(self.population) for i in range(self.tournament_size)]
                highest_fitness = -1
                highest_fit_individual = tournament[0]
                for individual in tournament:
                    if individual.fitness > highest_fitness:
                        highest_fitness = individual.fitness
                        highest_fit_individual = individual
                winners.append(highest_fit_individual)
            # Winners of the tournament (aka parents) are denoted tuples (parent_one, parent_two)
            # for the crossover operator.
            parent_pairs.append(tuple(winners))
        return parent_pairs

    # Returns a list of offspring from a lists of tuple parents. The offsprings are created via binomial crossover.
    def offspring(self, parent_pairs):
        offspring = []
        for parents in parent_pairs:
            offspring.extend(self.crossover(parents[0], parents[1]))
        return offspring

    # Generational Replacement: All members of the previous population are replaced by the offsprings.
    def replacement(self, offspring):
        self.population = offspring

    # Creates two individuals from two parents. Crossover occurs via a crossover probability of the parents. If
    # crossover does not occur, then the offsprings will have the same vector as the parents. Otherwise,
    # binomial crossover occurs.
    def crossover(self, parent_one, parent_two):
        child_a = []
        child_b = []
        for gene in range(len(parent_one.vector)):
            if flip(0.5):
                # Binomial crossover occurs. The children will both receive variations of genes of the parents.
                child_a.append(parent_one.vector[gene])
                child_b.append(parent_two.vector[gene])
            else:
                # Children will be the same as the parents
                child_a.append(parent_two.vector[gene])
                child_b.append(parent_one.vector[gene])
        return [Individual(self.network, np.asarray(child_a), self.mutation_prob, self.creep_variance),
                Individual(self.network, np.asarray(child_b), self.mutation_prob, self.creep_variance)]

    # Given a matrix of different shapes, it converts into a 1D numpy.array vector.
    def matrix_to_vector(self, matrix):
        vectors = []
        for weight in matrix:
            shape = weight.shape
            vectors.extend(weight.reshape(1, shape[0] * shape[1]).tolist())
        return numpy.array(list(itertools.chain.from_iterable(vectors)))

    # Given a numpy.array vector, a matrix is constructed with the appropriate shape that matches the network.
    def vector_to_matrix(self, vector):
        weights = []
        i = 0
        for shape in self.network.get_weight_shapes():
            size = reduce((lambda x, y: x * y), shape)
            weights.append(np.reshape(vector[i:i+size], shape))
            i += size
        return weights

    # Trains the neural network via genetic algorithm by constructing parents, reproduction of the offspring,
    # replacement of the entire population from the offspring, and finally updates the global best individual.
    def train(self):
        fitness_history = []
        # The fittest individual across all generations will be the individual selected to be the weights of our network
        global_best_individual = None
        global_best_fitness = 0
        while True:
            # Parent selection occurs.
            parents = self.parent_selection()
            # Offsprings are reproduced by the parents
            offspring = self.offspring(parents)
            # Previous population replaced by the offsprings
            self.replacement(offspring)
            # Record the fittest individual within that specific generation.
            highest_fitness = 0
            for individual in self.population:
                if individual.fitness > highest_fitness:
                    highest_fitness = individual.fitness
                # Record the global fittest individual as the best individual may not be at the end of the convergence
                # cycle. By recording the global best individual, we can use the weights of the best individual as
                # the fully trained weights in the neural network.
                if individual.fitness > global_best_fitness:
                    global_best_individual = individual
                    global_best_fitness = individual.fitness
            fitness_history.append(highest_fitness)
            # Convergence detection. If the improvement of the fitness does not significantly improve by our
            # predetermined threshold, we can terminate the genetic algorithm as it is deemed as trained.
            # Once convergence is reached and the algorithm has been terminated, the global fittest individual
            # will be the weights of the final neural network.
            if len(fitness_history) > self.convergence_size * 2:
                fitness_history.pop(0)
                older_fitness = sum(fitness_history[:self.convergence_size])
                newer_fitness = sum(fitness_history[self.convergence_size:])
                if newer_fitness <= older_fitness + CONVERGENCE_THRESHOLD:
                    # Convergence reached. Here, we set the network weights to be the global fittest individual.
                    # Terminates the genetic algorithm.
                    self.network.weights = self.vector_to_matrix(global_best_individual.vector)
                    return
