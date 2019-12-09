# genetic.py
# Implements genetic algorithm with neural networks

import random
import itertools
from functools import reduce

import math
import numpy
import numpy as np
import src.data as ds
from src.network import Network

# Search space for the neural network weights, tunable
LOWER_BOUND = -100000
UPPER_BOUND =  100000

# The amount of fitness increase used in the average metric to continue the learning process
CONVERGENCE_THRESHOLD = .0001

#
class Individual:
    def __init__(self, network: Network, vector, mutation_prob=0.0, creep_variance=1.0):
        self.network = network
        self.vector = vector
        for gene in range(len(self.vector)):
            if flip(mutation_prob):
                self.vector[gene] += np.random.normal(0, creep_variance)
        self.fitness = self.calc_fitness()

    def __str__(self):
        return "Vector: " + str(self.vector) + ", Fitness: " + str(self.fitness)


    def calc_fitness(self):
        weight_matrix = self.vector_to_matrix(self.vector)
        self.network.weights = weight_matrix
        if self.network.is_regression():
            fitness = 1 / self.network.get_error(self.network.training_data)
        else:
            fitness = self.network.get_accuracy(self.network.training_data)
        return fitness

    def vector_to_matrix(self, vector):
        weights = []
        i = 0
        for shape in self.network.get_weight_shapes():
            size = reduce((lambda x, y: x * y), shape)
            weights.append(np.reshape(vector[i:i+size], shape))
            i += size
        return weights


def flip(prob):
    return True if random.random() < prob else False


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
        self.population = [Individual(network, np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=network.get_num_weights())) for i in range(population_size)]

    def get_highest_fitness(self):
        max_fitness = 0
        max_fit_individual = self.population[0]
        for ind in self.population:
            if ind.fitness > max_fitness:
                max_fitness = ind.fitness
                max_fit_individual = ind
        return max_fit_individual.fitness

    def parent_selection(self):
        parent_pairs = []
        num_offsprings = int(self.population_size)
        for num in range(math.ceil(num_offsprings / 2)):
            winners = []
            for i in range(2):
                tournament = [random.choice(self.population) for i in range(self.tournament_size)]
                highest_fitness = -1
                highest_fit_individual = tournament[0]
                for individual in tournament:
                    if individual.fitness > highest_fitness:
                        highest_fitness = individual.fitness
                        highest_fit_individual = individual
                winners.append(highest_fit_individual)
            parent_pairs.append(tuple(winners))
        return parent_pairs

    def offspring(self, parent_pairs):
        offspring = []
        for parents in parent_pairs:
            offspring.extend(self.crossover(parents[0], parents[1]))
        return offspring

    def replacement(self, offspring):
        self.population = offspring

    def crossover(self, parent_one, parent_two):
        child_a = []
        child_b = []
        for gene in range(len(parent_one.vector)):
            if flip(0.5):
                child_a.append(parent_one.vector[gene])
                child_b.append(parent_two.vector[gene])
            else:
                child_a.append(parent_two.vector[gene])
                child_b.append(parent_one.vector[gene])
        return [Individual(self.network, np.asarray(child_a), self.mutation_prob, self.creep_variance),
                Individual(self.network, np.asarray(child_b), self.mutation_prob, self.creep_variance)]


    def calc_fitness(self, weight_matrix):
        self.network.weights = weight_matrix
        if self.network.is_regression():
            fitness = 1 / self.network.get_error(self.network.training_data)
        else:
            fitness = self.network.get_accuracy(self.network.training_data)
        return fitness

    def matrix_to_vector(self, matrix):
        vectors = []
        for weight in matrix:
            shape = weight.shape
            vectors.extend(weight.reshape(1, shape[0] * shape[1]).tolist())
        return numpy.array(list(itertools.chain.from_iterable(vectors)))

    def vector_to_matrix(self, vector):
        weights = []
        i = 0
        for shape in self.network.get_weight_shapes():
            size = reduce((lambda x, y: x * y), shape)
            weights.append(np.reshape(vector[i:i+size], shape))
            i += size
        return weights

    def train(self):
        fitness_history = []
        global_best_individual = None
        global_best_fitness = 0
        while True:
            parents = self.parent_selection()
            offspring = self.offspring(parents)
            self.replacement(offspring)
            highest_fitness = 0
            for individual in self.population:
                if individual.fitness > highest_fitness:
                    highest_fitness = individual.fitness

                if individual.fitness > global_best_fitness:
                    global_best_individual = individual
                    global_best_fitness = individual.fitness
            print("GLOBAL BEST " + str(global_best_fitness))
            fitness_history.append(highest_fitness)
            # print("HIGHEST IN GENERATION: " + str(highest_fitness))
            if len(fitness_history) > self.convergence_size * 2:
                fitness_history.pop(0)
                older_fitness = sum(fitness_history[:self.convergence_size])
                newer_fitness = sum(fitness_history[self.convergence_size:])
                if newer_fitness <= older_fitness + CONVERGENCE_THRESHOLD:
                    self.network.weights = self.vector_to_matrix(global_best_individual.vector)
                    return


def test_genetic_machine():
    population_size = 100
    crossover_prob = 0.5
    creep = 1
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 100

    data = ds.get_machine_data("../../data/machine.data")
    training, test = data.partition(.9)
    net = Network(training, test, [6, 2, 1])
    ga = Genetic(net, population_size, crossover_prob, creep, mutation_prob, tournament_k, convergence_size)
    ga.train()


def test_genetic_car():
    population_size = 100
    crossover_prob = 0.5
    creep = 1000
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 100

    data = ds.get_car_data("../../data/car.data")
    training, test = data.partition(.9)
    net = Network(training, test, [6, 5, 4], ["acc", "unacc", "good", "vgood"])
    ga = Genetic(net, population_size, crossover_prob, creep, mutation_prob, tournament_k, convergence_size)
    ga.train()

def test_genetic_image():
    population_size = 100
    crossover_prob = 0.5
    creep = 1
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 100

    data = ds.get_segmentation_data("../data/segmentation.data")
    training, test = data.partition(.9)
    net = Network(training, test, [19, 13, 7], ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"])
    ga = Genetic(net, population_size=population_size, crossover_prob=crossover_prob, creep_variance=creep, mutation_prob=mutation_prob, tournament_size=tournament_k, convergence_size=convergence_size)
    ga.train()









