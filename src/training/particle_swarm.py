# particle_swarm.py
# Implementation of particle swarm optimization (PSO).

import random
import numpy as np
import src.data as ds
from src.network import Network
from functools import reduce


class ParticleSwarm:

    def __init__(self, network: Network, pop_size: int, cog_factor: float, soc_factor: float, inertia: float, max_velocity: float, convergence_size: int):
        self.network = network
        self.pop_size = pop_size
        self.cog_factor = cog_factor
        self.soc_factor = soc_factor
        self.inertia = inertia
        self.max_velocity = max_velocity
        self.convergence_size = convergence_size
        self.particles = [Particle(network, max_velocity) for i in range(pop_size)]

    def __get_global_best(self):
        most_fit_particle = self.particles[0]
        highest_fitness = self.particles[0].get_fitness()
        for particle in self.particles[1:]:
            particle_fitness = particle.get_fitness()
            if particle_fitness > highest_fitness:
                most_fit_particle = particle
                highest_fitness = particle_fitness
        return most_fit_particle

    def train(self):
        for i in range(100):
            g_best = self.__get_global_best()
            print(g_best.get_fitness())
            for particle in self.particles:
                particle.move()
                particle.update_velocity(self.cog_factor, self.soc_factor, self.inertia, g_best.state)
        pass


LOWER_BOUND = -100000
UPPER_BOUND = 100000


class Particle:

    def __init__(self, network: Network, max_velocity: float):
        self.network = network
        self.data_set = network.training_data
        self.max_velocity = max_velocity
        self.state = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=network.get_num_weights())
        self.velocity = np.random.uniform(low=-1, high=1, size=network.get_num_weights())
        self.velocity = self.velocity * (max_velocity / np.linalg.norm(self.velocity))
        self.p_best = self.state
        self.best_fitness = self.get_fitness()

    def __str__(self):
        return "Particle{state: " + str(self.state) + ",\nvelocity: " + str(self.velocity) + "}"

    def encode(self, weight_shapes: list):
        weights = []
        i = 0
        for shape in weight_shapes:
            size = reduce((lambda x, y: x * y), shape)
            weights.append(np.reshape(self.state[i:i+size], shape))
            i += size
        return weights

    def get_fitness(self):
        old_weights = self.network.weights
        self.network.weights = self.encode(self.network.get_weight_shapes())
        if self.network.is_regression():
            fitness = 1/self.network.get_error(self.data_set)
        else:
            fitness = self.network.get_accuracy(self.data_set)
        self.network.weights = old_weights
        return fitness

    def move(self):
        # print("velocity=" + str(np.linalg.norm(self.velocity)))
        self.state += self.velocity
        new_fitness = self.get_fitness()
        if new_fitness > self.best_fitness:
            self.p_best = self.state
            self.best_fitness = new_fitness

    def update_velocity(self, cog_factor: float, soc_factor: float, inertia: float, g_best: np.ndarray):
        cognitive = random.random() * cog_factor * (self.p_best - self.state)
        social = random.random() * soc_factor * (g_best - self.state)
        self.velocity = self.velocity * inertia + cognitive + social
        # Make sure velocity does not exceed maximum:
        magnitude = np.linalg.norm(self.velocity)
        self.velocity = self.velocity * (self.max_velocity/magnitude)


def test_particle_swarm_car():
    data = ds.get_car_data("../../data/car.data")
    training, test = data.partition(.9)
    net = Network(training, test, [6, 5, 4], ["acc", "unacc", "good", "vgood"])
    # a = Particle(22, 10.5)
    # print(a.state)
    # [print(weight) for weight in a.encode([(3,4), (2, 3), (2, 2)])]
    b = ParticleSwarm(net, pop_size=100, cog_factor=0.01, soc_factor=0.05, inertia=0.05, max_velocity=100000, convergence_size=100)
    b.train()

def test_particle_swarm_machine():
    data = ds.get_machine_data("../../data/machine.data")
    training, test = data.partition(.9)
    net = Network(training, test, [6, 3, 1])

    b = ParticleSwarm(net, pop_size=100, cog_factor=0.01, soc_factor=0.05, inertia=0.05, max_velocity=100000, convergence_size=100)
    b.train()

# test_particle_swarm_car()
test_particle_swarm_machine()