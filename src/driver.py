from src.training.particle_swarm import ParticleSwarm
import src.data as data
from src.network import Network
from src.training.diff_evolution import DiffEvolution
from  src.training.genetic import Genetic


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


# Runs PSO on a classification data set using 10-fold cross validation.
def classification_diff_evol(data_set, data_set_name, classes, mutationF, recombinationC, pop_size):
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
            diffevo = DiffEvolution(network, mutationF, recombinationC, pop_size)
            diffevo.run()

            accuracy = network.get_accuracy(test)
            average_accuracy += accuracy / 10
            print("----Accuracy of fold {}: {:.2f}".format(fold_i, accuracy))
        print("--Final accuracy: {:.2f}".format(average_accuracy))


def regression_diff_evol(data_set, data_set_name, mutationF, recombinationC, pop_size):
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
            diffevo = DiffEvolution(network, mutationF, recombinationC, pop_size)
            diffevo.run()

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

    # classification_particle_swarm(abalone_data, "abalone.data", abalone_data_classes,
    #                               pop_size=50,
    #                               cog_factor=0.1,
    #                               soc_factor=0.07,
    #                               inertia=0.01,
    #                               max_velocity=100000,
    #                               convergence_size=20)
    #
    # classification_particle_swarm(car_data, "car.data", ["acc", "unacc", "good", "vgood"],
    #                               pop_size=50,
    #                               cog_factor=0.1,
    #                               soc_factor=0.07,
    #                               inertia=0.01,
    #                               max_velocity=1000000,
    #                               convergence_size=20)
    #
    # classification_particle_swarm(segmentation_data, "segmentation.data", segmentation_classes,
    #                               pop_size=50,
    #                               cog_factor=0.1,
    #                               soc_factor=0.07,
    #                               max_velocity=100000,
    #                               convergence_size=20)

    # regression_particle_swarm(machine_data, "machine.data",
    #                           pop_size=100,
    #                           cog_factor=0.2,
    #                           soc_factor=0.1,
    #                           inertia=0.05,
    #                           max_velocity=100000,
    #                           convergence_size=20)
    #
    # classification_diff_evol(machine_data, "machine.data", ["acc", "unacc", "good", "vgood"],
    #                  mutationF=.1,
    #                  recombinationC=.9,
    #                  pop_size=40,
    #                  )

    # regression_diff_evol(machine_data, "machine.data",
    #                      mutationF=.1,
    #                      recombinationC=.9,
    #                      pop_size=40,
    #                      )

    classification_genetic(car_data, "car.data", car_data_classes,
                           population_size=100,
                           crossover_prob=0.5,
                           creep_variance=100,
                           mutation_prob=0.05,
                           tournament_size=2,
                           convergence_size=100)

    classification_genetic(abalone_data, "abalone.data", abalone_data_classes,
                           population_size=100,
                           crossover_prob=0.5,
                           creep_variance=100,
                           mutation_prob=0.05,
                           tournament_size=2,
                           convergence_size=100)

    classification_genetic(segmentation_data, "image-segmentation.data", segmentation_classes,
                           population_size=100,
                           crossover_prob=0.5,
                           creep_variance=100,
                           mutation_prob=0.05,
                           tournament_size=2,
                           convergence_size=100)

    regression_genetic(machine_data, "machine.data",
                       population_size=100,
                       crossover_prob=0.5,
                       creep_variance=1,
                       mutation_prob=0.05,
                       tournament_size=2,
                       convergence_size=100)

    regression_genetic(forest_fire_data, "forest_fire.data",
                       population_size=100,
                       crossover_prob=0.5,
                       creep_variance=1,
                       mutation_prob=0.05,
                       tournament_size=2,
                       convergence_size=100)

    regression_genetic(wine_data, "wine.data",
                       population_size=100,
                       crossover_prob=0.5,
                       creep_variance=1,
                       mutation_prob=0.05,
                       tournament_size=2,
                       convergence_size=100)


main()
