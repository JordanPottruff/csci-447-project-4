import random
import itertools
from functools import reduce

import numpy
import numpy as np
import src.data as ds
from src.network import Network

LOWER_BOUND = -100000
UPPER_BOUND =  100000


def flip(prob):
    return True if random.random() < prob else False


class Genetic:
    def __init__(self, network: Network, population_size: int, crossover_prob: float, steady_state_perc: float,
            creep_variance: float, mutation_prob: float, tournament_size: int):
        self.network = network
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.steady_state_perc = steady_state_perc
        self.creep_variance = creep_variance
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.population = self.create_population()

    # Population holds individuals denoted by (weight matrix, fitness)
    def create_population(self):
        population = []
        for i in range(self.population_size):
            individual = []
            for shape in self.network.get_weight_shapes():
                weight_matrix = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=shape)
                individual.append(weight_matrix)
            fitness = self.calc_fitness(individual)
            population.append((individual, fitness))
        return population

    # TODO: Finish Selection
    def selection(self):
        parents = []
        parents_index = []
        for i in range(2):
            tournament = [random.choice(range(len(self.population))) for i in range(self.tournament_size)]
            highest_fitness = -1
            highest_fit_individual = None
            index = None
            for j in tournament:
                if self.population[j][1] > highest_fitness:
                    highest_fitness = self.population[j][1]
                    highest_fit_individual = self.population[j]
                    index = j
            parents.append(self.population[j])
            parents_index.append(j)

        if flip(self.crossover_prob):
            self.crossover(parents[0][0], parents[1][0])

    def crossover(self, parent_one, parent_two):
        parent_one = self.matrix_to_vector(parent_one)
        parent_two = self.matrix_to_vector(parent_two)
        for chromosomes in range(len(parent_one)):
            if flip(0.5):
                temp = parent_one[chromosomes]
                parent_one[chromosomes] = parent_two[chromosomes]
                parent_two[chromosomes] = temp
        return parent_one, parent_two

    def mutation(self, chromosome):
        if flip(self.mutation_prob):
            index = random.choice(range(len(chromosome)))
            chromosome[index] = chromosome[index] + np.random.normal(0, 1)
        return chromosome


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


def test_genetic():
    data = ds.get_machine_data("../../data/machine.data")
    training, test = data.partition(.9)
    net = Network(training, test, [6, 3, 1])
    ga = Genetic(net, 10, .8, .8, .8, .1, 2)
    ga.selection()


test_genetic()




