# Driver for project video.
# Executes all algorithms, displaying intermediate steps for each. We tuned the parameters for fast convergence rather
# than fitness to expedite the runtime of the algorithm.

import src.data as data
from src.video.particle_swarm_video import ParticleSwarm
from src.video.genetic_video import Genetic
from src.video.diff_evolution_video import DiffEvolution
from src.network import Network


# Runs the PSO algorithm on the image data set. Displays the state, fitness, particle, and weights as the
# intermediate steps.
def test_particle_swarm_image():
    image_data = data.get_segmentation_data("../../data/segmentation.data")
    training_data, test_data = image_data.partition(.8)
    network = Network(training_data, test_data, [19, 13, 7], ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT",
                                                              "WINDOW", "PATH", "GRASS"])

    pso = ParticleSwarm(network, pop_size=20, cog_factor=1.0, soc_factor=2.0, inertia=0.05,
                        max_velocity=100000, convergence_size=50)
    pso.train()

    accuracy = network.get_accuracy(test_data)*100
    print("\n\nAccuracy on test set: {}%".format(accuracy))


# The following 6 function executes genetic algorithms on all data sets. For the genetic algorithm, we display
# the fitness and the vector as the intermediate steps.
def test_genetic_machine():
    population_size = 20
    crossover_prob = 0.5
    creep = 1
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 100

    machine_data = data.get_machine_data("../../data/machine.data")
    training, test = machine_data.partition(.9)
    network = Network(training, test, [6, 3, 1])
    ga = Genetic(network, population_size, crossover_prob, creep, mutation_prob, tournament_k, convergence_size)
    ga.train()

    error = network.get_error(test) * 100
    print("\n\nError on test set: {}%".format(error))


def test_genetic_forest_fires():
    population_size = 20
    crossover_prob = 0.5
    creep = 1
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 100

    forest_fire_data = data.get_forest_fire_data("../../data/forestfires.data")
    training, test = forest_fire_data.partition(.9)
    network = Network(training, test, [12, 6, 1])
    ga = Genetic(network, population_size, crossover_prob, creep, mutation_prob, tournament_k, convergence_size)
    ga.train()

    error = network.get_error(test) * 100
    print("\n\nError on test set: {}%".format(error))


def test_genetic_wine():
    population_size = 20
    crossover_prob = 0.5
    creep = 1
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 100

    wine_data = data.get_wine_data("../../data/winequality.data")
    training, test = wine_data.partition(.9)
    network = Network(training, test, [11, 6, 1])
    ga = Genetic(network, population_size, crossover_prob, creep, mutation_prob, tournament_k, convergence_size)
    ga.train()

    error = network.get_error(test) * 100
    print("\n\nError on test set: {}%".format(error))


def test_genetic_image():
    population_size = 20
    crossover_prob = 0.5
    creep = 20
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 50

    image_data = data.get_segmentation_data("../../data/segmentation.data")
    training_data, testing_data = image_data.partition(0.8)
    network = Network(training_data, testing_data, [19, 13, 7], ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT",
                                                                 "WINDOW", "PATH", "GRASS"])
    ga = Genetic(network, population_size, crossover_prob, creep, mutation_prob, tournament_k, convergence_size)
    ga.train()
    accuracy = network.get_accuracy(testing_data)*100
    print("\n\nAccuracy on test set: {}%".format(accuracy))


def test_genetic_car():
    population_size = 20
    crossover_prob = 0.5
    creep = 20
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 50

    car_data = data.get_car_data("../../data/car.data")
    training_data, testing_data = car_data.partition(0.8)
    network = Network(training_data, testing_data, [6, 5, 4], ["acc", "unacc", "good", "vgood"])
    ga = Genetic(network, population_size, crossover_prob, creep, mutation_prob, tournament_k, convergence_size)
    ga.train()
    accuracy = network.get_accuracy(testing_data)*100
    print("\n\nAccuracy on test set: {}%".format(accuracy))


def test_genetic_abalone():
    population_size = 20
    crossover_prob = 0.5
    creep = 1
    mutation_prob = 0.05
    tournament_k = 2
    convergence_size = 100

    abalone_data = data.get_abalone_data("../../data/abalone.data")
    training_data, testing_data = abalone_data.partition(0.8)
    network = Network(training_data, testing_data, [7, 4, 1], [i for i in range(1, 30)])
    ga = Genetic(network, population_size, crossover_prob, creep, mutation_prob, tournament_k, convergence_size)
    ga.train()

    accuracy = network.get_accuracy(testing_data) * 100
    print("\n\nAccuracy on test set: {}%".format(accuracy))


# Executes differential evolution algorithm on the image data. We display the mutant, individual, after recomination
# vectors, and the fitness for each individual as the intermediate steps.
def test_diff_evolution_image():
    image_data = data.get_segmentation_data("../../data/segmentation.data")
    training_data, test_data = image_data.partition(.8)
    network = Network(training_data, test_data, [19, 13, 7], ["BRICKFACE", "SKY", "FOLIAGE",
                                                              "CEMENT", "WINDOW", "PATH", "GRASS"])

    diff_evo = DiffEvolution(network, mutation_f=.1, recombination_c=.9, pop_size=20)
    diff_evo.run()

    accuracy = network.get_accuracy(test_data)*100
    print("\n\nAccuracy on test set: {}%".format(accuracy))

# test_diff_evolution_image()
# test_particle_swarm_image()

# Video Files for the GA: car.data (classification) and machine.data (regression) executes the fastest, on average.
# test_genetic_image()
# test_genetic_car()
# test_genetic_abalone()
# test_genetic_forest_fires()
# test_genetic_wine()
# test_genetic_machine()
