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

    # Returns the particle with the current best state among all the particles.
    def __get_global_best(self):
        most_fit_particle = self.particles[0]
        highest_fitness = self.particles[0].get_fitness()
        for particle in self.particles[1:]:
            particle_fitness = particle.get_fitness()
            if particle_fitness > highest_fitness:
                most_fit_particle = particle
                highest_fitness = particle_fitness
        return most_fit_particle

    # Begins the training process. Moves each particle according to its velocity, and then updates the velocity of each
    # particle.
    # TODO: train needs to update the network's weights with the best weights found.
    # TODO: add convergence check (use convergence_size parameter) similar to backprop.
    def train(self):
        for i in range(100):
            g_best = self.__get_global_best()
            print(g_best.get_fitness())
            for particle in self.particles:
                particle.move()
                particle.update_velocity(self.cog_factor, self.soc_factor, self.inertia, g_best.state)


LOWER_BOUND = -100000
UPPER_BOUND = 100000


# Particle class that contains a lot of the central logic for the particle swarm technique.
class Particle:

    # Creates a new particle. Requires a network object in order to generate the right (random) vector representing the
    # weights. Also requires a max velocity parameter that will limit how fast a particle moves through the state space.
    def __init__(self, network: Network, max_velocity: float):
        self.network = network
        self.max_velocity = max_velocity
        self.state = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=network.get_num_weights())
        self.velocity = np.random.uniform(low=-1, high=1, size=network.get_num_weights())
        self.velocity = self.velocity * (max_velocity / np.linalg.norm(self.velocity))
        self.p_best = self.state
        self.best_fitness = self.get_fitness()

    # Returns a string representation of the particle, showing the state and velocity.
    def __str__(self):
        return "Particle{state: " + str(self.state) + ",\nvelocity: " + str(self.velocity) + "}"

    # Turns the current particle into a list of weight matrices to use in a network.
    def encode(self):
        weights = []
        i = 0
        for shape in self.network.get_weight_shapes():
            size = reduce((lambda x, y: x * y), shape)
            weights.append(np.reshape(self.state[i:i+size], shape))
            i += size
        return weights

    # Returns the fitness of the particle's current state. The fitness is evaluated on the training set for the network.
    # The fitness is accuracy if a classification problem, and the inverse error for regression.
    def get_fitness(self):
        old_weights = self.network.weights
        self.network.weights = self.encode()
        if self.network.is_regression():
            fitness = 1/self.network.get_error(self.network.training_data)
        else:
            fitness = self.network.get_accuracy(self.network.training_data)
        self.network.weights = old_weights
        return fitness

    # Moves the particle from the current state to a new state based on the current velocity.
    def move(self):
        # print("velocity=" + str(np.linalg.norm(self.velocity)))
        self.state += self.velocity
        new_fitness = self.get_fitness()
        if new_fitness > self.best_fitness:
            self.p_best = self.state
            self.best_fitness = new_fitness

    # Updates the velocity based on the gbest particle swarm technique.
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