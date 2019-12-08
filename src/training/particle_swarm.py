# particle_swarm.py
# Implementation of particle swarm optimization (PSO).

import random
import numpy as np
import src.data as ds
from src.network import Network
from functools import reduce

# The amount each new average metric needs to be better than the old average metric for the training process to
# continue.
CONVERGENCE_THRESHOLD = .0001


# Used to create an instance of the particle swarm optimization algorithm. Trains a neural network according to the
# movement of particles through the state space (i.e. the possible weights in the network), evaluating fitness on the
# network's training set.
class ParticleSwarm:

    # Creates an instance of the particle swarm algorithm. The particle swarm trains the given network according to the
    # given parameters.
    # * network: the network object to train.
    # * pop_size: the size of the population; the number of particles.
    # * cog_factor: the coefficient influencing how much to bias velocity towards a particle's personal best state.
    # * soc_factor: the coefficient influencing how much to bias velocity towards the global best state.
    # * max_velocity: the maximum velocity possible, ensuring no velocity vector has a magnitude larger than this.
    # * convergence_size: how many items to evaluate when determining whether convergence has occurred.
    # Creating the ParticleSwarm strategy does not train the network. To train the network, the train() method must be
    # called.
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
    def train(self):
        # Fitness history will be a list of the global best fitness of each run/
        fitness_history = []
        # We will need to store the best_particle to return when the algorithm has converged. This is distinct from the
        # "global best" particle, because it is possible that our final loop results in a global best that is less fit
        # compared to the global best of a different iteration of the while loop. So we must store the best we have ever
        # seen across all runs.
        best_particle, best_particle_fitness = (None, float("-inf"))

        while True:
            print(best_particle_fitness)
            # (1) Determine the global best particle (based on fitness), store as g_best.
            g_best = self.__get_global_best()
            g_best_fitness = g_best.get_fitness()

            # (2) If the current global best is better than any we have seen, update best_particle.
            if g_best_fitness > best_particle_fitness:
                best_particle, best_particle_fitness = g_best, g_best_fitness

            # (3) Determine if we have reached convergence by evaluating the fitness of the last n runs and comparing it
            # to the n runs before that, where n is self.convergence_size. If converged, update then network with the
            # best weights we have ever seen.
            fitness_history.append(g_best.get_fitness())
            # Only do the convergence check if we have enough runs (i.e. 2 * self.convergence_size).
            if len(fitness_history) > self.convergence_size * 2:
                fitness_history.pop(0)
                older_fitness = sum(fitness_history[:self.convergence_size])
                newer_fitness = sum(fitness_history[self.convergence_size:])
                # Exit if our fitness has decreased over time, with some added threshold to prevent extremely small
                # gains from preventing the algorithm from exiting.
                if newer_fitness <= older_fitness + CONVERGENCE_THRESHOLD:
                    self.network.weights = best_particle.encode()
                    return

            # (4) Finally, we can go through the particles and update them. This involves moving them according to their
            # current velocity and then updating their velocity according to the global best and the tunable parameters.
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
# test_particle_swarm_machine()