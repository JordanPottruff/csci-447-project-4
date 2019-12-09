# driver.py
# Entry file for executing our experiment design for PSO, Genetic. and Differential Evolution algorithms.
# The driver contains function to run regression and classification data sets per each algorithm.

from src.training.particle_swarm import ParticleSwarm
import src.data as data
from src.network import Network
from src.training.diff_evolution import DiffEvolution
from src.training.genetic import Genetic


# Tuned parameter for number of hidden layers (0, 1, and 2) and the number of nodes per each hidden layer.
# The number of nodes in the hidden layers is calculated by the total number of node in the input and output dividing
# by two, rounding down.
def get_network_layouts(num_in, num_out):
    avg_in_out = (num_in + num_out) // 2

    zero_layer_nn = [num_in, num_out]
    one_layer_nn = [num_in, avg_in_out, num_out]
    two_layer_nn = [num_in, avg_in_out, avg_in_out, num_out]
    return [zero_layer_nn, one_layer_nn, two_layer_nn]


# Runs PSO on a classification data set using 10-fold cross validation.
def classification_particle_swarm(data_set, data_set_name, classes, pop_size, cog_factor, soc_factor, inertia, max_velocity, convergence_size):
    print("Running classification on: {}".format(data_set_name))
    network_layouts = get_network_layouts(data_set.num_cols, len(classes))

    folds = data_set.validation_folds(10)
    for layer_sizes in network_layouts:
        average_accuracy = 0
        print("--Testing network layout: {}".format(layer_sizes))
        for fold_i, fold in enumerate(folds):
            train = fold['train']
            test = fold['test']

            network = Network(train, test, layer_sizes, classes)
            pso = ParticleSwarm(network, pop_size, cog_factor, soc_factor, inertia, max_velocity, convergence_size)
            pso.train()

            accuracy = network.get_accuracy(test)
            average_accuracy += accuracy / 10
            print("----Accuracy of fold {}: {:.2f}".format(fold_i, accuracy))
        print("--Final accuracy: {:.2f}".format(average_accuracy))


# Runs the PSO algorithm on a regression data set using 10 folds cross validation.
def regression_particle_swarm(data_set, data_set_name, pop_size, cog_factor, soc_factor, inertia, max_velocity, convergence_size):
    print("Running regression on: {}".format(data_set_name))
    network_layouts = get_network_layouts(data_set.num_cols, 1)

    folds = data_set.validation_folds(10)
    for layer_sizes in network_layouts:
        average_error = 0
        print("--Testing network layout: {}".format(layer_sizes))
        for fold_i, fold in enumerate(folds):
            train = fold['train']
            test = fold['test']

            network = Network(train, test, layer_sizes)
            pso = ParticleSwarm(network, pop_size, cog_factor, soc_factor, inertia, max_velocity, convergence_size)
            pso.train()

            error = network.get_error(test)
            average_error += error / 10
            print("----Error of fold {}: {:.2f}".format(fold_i, error))
        print("--Final error: {:.2f}".format(average_error))


# Runs differential evolution algorithm on a classification data set using 10-fold cross validation.
def classification_diff_evolution(data_set, data_set_name, classes, mutation_f, recombination_c, pop_size):
    print("Running classification on: {}".format(data_set_name))
    network_layouts = get_network_layouts(data_set.num_cols, len(classes))

    folds = data_set.validation_folds(10)
    for layer_sizes in network_layouts:
        average_accuracy = 0
        print("--Testing network layout: {}".format(layer_sizes))
        for fold_i, fold in enumerate(folds):
            train = fold['train']
            test = fold['test']

            network = Network(train, test, layer_sizes, classes)
            diff_evolution = DiffEvolution(network, mutation_f, recombination_c, pop_size)
            diff_evolution.run()

            accuracy = network.get_accuracy(test)
            average_accuracy += accuracy / 10
            print("----Accuracy of fold {}: {:.2f}".format(fold_i, accuracy))
        print("--Final accuracy: {:.2f}".format(average_accuracy))


# Runs the differential evolution algorithm on a regression data set using 10-fold cross validation.
def regression_diff_evolution(data_set, data_set_name, mutation_f, recombination_c, pop_size):
    print("Running regression on: {}".format(data_set_name))
    network_layouts = get_network_layouts(data_set.num_cols, 1)

    folds = data_set.validation_folds(10)
    for layer_sizes in network_layouts:
        average_error = 0
        print("--Testing network layout: {}".format(layer_sizes))
        for fold_i, fold in enumerate(folds):
            train = fold['train']
            test = fold['test']

            network = Network(train, test, layer_sizes)
            diff_evolution = DiffEvolution(network, mutation_f, recombination_c, pop_size)
            diff_evolution.run()

            error = network.get_error(test)
            average_error += error / 10
            print("----Error of fold {}: {:.2f}".format(fold_i, error))
        print("--Final error: {:.2f}".format(average_error))


# Runs genetic algorithm on a classification data set using 10-fold cross validation.
def classification_genetic(data_set, data_set_name, classes, population_size, crossover_prob, creep_variance,
                           mutation_prob, tournament_size, convergence_size):
    print("Running classification on: {}".format(data_set_name))
    network_layouts = get_network_layouts(data_set.num_cols, len(classes))

    folds = data_set.validation_folds(10)
    for layer_sizes in network_layouts:
        average_accuracy = 0
        print("--Testing network layout: {}".format(layer_sizes))
        for fold_i, fold in enumerate(folds):
            train = fold['train']
            test = fold['test']

            network = Network(train, test, layer_sizes, classes)
            ga = Genetic(network, population_size=population_size, crossover_prob=crossover_prob,
                         creep_variance=creep_variance, mutation_prob=mutation_prob,
                         tournament_size=tournament_size, convergence_size=convergence_size)
            ga.train()

            accuracy = network.get_accuracy(test)
            average_accuracy += accuracy / 10
            print("----Accuracy of fold {}: {:.2f}".format(fold_i, accuracy))
        print("--Final accuracy: {:.2f}".format(average_accuracy))


# Runs genetic algorithm on a regression data set using 10-folds cross validation.
def regression_genetic(data_set, data_set_name, population_size, crossover_prob, creep_variance,
                       mutation_prob, tournament_size, convergence_size):
    print("Running regression on: {}".format(data_set_name))
    network_layouts = get_network_layouts(data_set.num_cols, 1)

    folds = data_set.validation_folds(10)
    for layer_sizes in network_layouts:
        average_error = 0
        print("--Testing network layout: {}".format(layer_sizes))
        for fold_i, fold in enumerate(folds):
            train = fold['train']
            test = fold['test']

            network = Network(train, test, layer_sizes)
            ga = Genetic(network, population_size, crossover_prob, creep_variance, mutation_prob,
                         tournament_size, convergence_size)

            ga.train()

            error = network.get_error(test)
            average_error += error / 10
            print("----Error of fold {}: {:.2f}".format(fold_i, error))
        print("--Final error: {:.2f}".format(average_error))


def main():
    # Classification data sets.
    abalone_data = data.get_abalone_data()
    car_data = data.get_car_data()
    segmentation_data = data.get_segmentation_data()

    # Classes for classification data sets.
    abalone_data_classes = [float(i) for i in range(1, 30)]
    car_data_classes = ["acc", "unacc", "good", "vgood"]
    segmentation_classes = ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"]

    # Regression data sets.
    forest_fire_data = data.get_forest_fire_data()
    machine_data = data.get_machine_data()
    wine_data = data.get_wine_data()

    # Tunable parameter for the genetic algorithm.
    population_size = 20
    crossover_probability = 0.5
    creep = 100
    mutation_prop = 0.05
    tournament_size = 2
    convergence_size = 50

    # Entry main function for running our experiments with all 3 algorithms: GA, DiffE, and PSO.

    classification_particle_swarm(abalone_data, "abalone.data", abalone_data_classes,
                                  pop_size=50,
                                  cog_factor=0.1,
                                  soc_factor=0.07,
                                  inertia=0.01,
                                  max_velocity=100000,
                                  convergence_size=20)

    classification_particle_swarm(car_data, "car.data", ["acc", "unacc", "good", "vgood"],
                                  pop_size=50,
                                  cog_factor=0.1,
                                  soc_factor=0.07,
                                  inertia=0.01,
                                  max_velocity=1000000,
                                  convergence_size=20)

    classification_particle_swarm(segmentation_data, "segmentation.data", segmentation_classes,
                                  pop_size=50,
                                  cog_factor=0.1,
                                  soc_factor=0.07,
                                  max_velocity=100000,
                                  convergence_size=20)

    regression_particle_swarm(machine_data, "machine.data",
                              pop_size=100,
                              cog_factor=0.2,
                              soc_factor=0.1,
                              inertia=0.05,
                              max_velocity=100000,
                              convergence_size=20)

    classification_genetic(car_data, "car.data", car_data_classes,
                           population_size=population_size,
                           crossover_prob=crossover_probability,
                           creep_variance=creep,
                           mutation_prob=mutation_prop,
                           tournament_size=tournament_size,
                           convergence_size=convergence_size)

    classification_genetic(abalone_data, "abalone.data", abalone_data_classes,
                           population_size=population_size,
                           crossover_prob=crossover_probability,
                           creep_variance=creep,
                           mutation_prob=mutation_prop,
                           tournament_size=tournament_size,
                           convergence_size=convergence_size)

    classification_genetic(segmentation_data, "image-segmentation.data", segmentation_classes,
                           population_size=population_size,
                           crossover_prob=crossover_probability,
                           creep_variance=creep,
                           mutation_prob=mutation_prop,
                           tournament_size=tournament_size,
                           convergence_size=convergence_size)

    regression_genetic(machine_data, "machine.data",
                       population_size=population_size,
                       crossover_prob=crossover_probability,
                       creep_variance=creep,
                       mutation_prob=mutation_prop,
                       tournament_size=tournament_size,
                       convergence_size=convergence_size)

    regression_genetic(forest_fire_data, "forest_fire.data",
                       population_size=population_size,
                       crossover_prob=crossover_probability,
                       creep_variance=creep,
                       mutation_prob=mutation_prop,
                       tournament_size=tournament_size,
                       convergence_size=convergence_size)

    regression_genetic(wine_data, "wine.data",
                       population_size=population_size,
                       crossover_prob=crossover_probability,
                       creep_variance=creep,
                       mutation_prob=mutation_prop,
                       tournament_size=tournament_size,
                       convergence_size=convergence_size)

    classification_diff_evolution(machine_data, "machine.data", ["acc", "unacc", "good", "vgood"],
                                  mutation_f=.1,recombination_c=.9,pop_size=40)

    regression_diff_evolution(machine_data, "machine.data",
                              mutation_f=.1,
                              recombination_c=.9,
                              pop_size=40,
                              )


main()
