# network.py
# Defines an instance of a neural network.

import math
import numpy as np
import src.util.activations as af
from src.data import DataSet


# The network class is meant to represent the functioning of a neural network.
class Network:

    # Creates a new neural network.
    # * training_data: the DataSet for the training observations.
    # * test_data: the DataSet for testing observations.
    # * layer_sizes: a list of sizes of each layer, with the first value being the number of nodes in the input layer
    #   and the last value being the number of nodes in the output layer. The middle values represent the number of
    #   hidden nodes in successive hidden layers.
    # * classes: for classification problems, this is a list of class names for the given data.
    def __init__(self, training_data: DataSet, test_data: DataSet, layer_sizes: list, classes=None):
        self.training_data = training_data
        self.test_data = test_data
        self.layer_sizes = layer_sizes
        self.classes = classes
        self.weights = [random_weight_matrix(layer_sizes[i-1]+1, layer_sizes[i]) for i in range(1, len(layer_sizes))]

    # Returns a string representation of the network.
    def __str__(self):
        return str(self.weights)

    # Returns true if the current problem is a regression problem. Otherwise, returns false if a classification problem.
    def is_regression(self):
        return self.classes is None

    # Given an expected output value (either a class or regression output), returns the vector form of this expected
    # output. In a regression, the output value is simply put into a vector of size 1. If a classification, the vector
    # is the length of the number of classes, with all values set to zero except for the value corresponding to the
    # given class (output_value).
    def expected_output_vector(self, output_value):
        if self.is_regression():
            return np.array([output_value])
        else:
            # Initialize all vectors with zero values.
            output_vector = np.zeros(len(self.classes))
            # Replace the value at the index for the class with a 1.
            class_index = self.classes.index(output_value)
            output_vector[class_index] = 1
            return output_vector

    # Returns a list of "shapes" of each weight matrix. Shapes are tuples in the form (row, col).
    def get_weight_shapes(self):
        return [weight.shape for weight in self.weights]

    # Returns the number of weights in the network. Includes weights from bias nodes.
    def get_num_weights(self):
        return sum([weight.size for weight in self.weights])

    # Gets the activations of the network caused by a given observation.
    # * observation: a numpy array representing the attribute values of a single observation.
    # Returns a list of numpy arrays, with the 0th position being the activation of the input layer, and so on.
    def get_activation(self, observation):
        inputs = np.append(observation, af.sigmoid(1))
        activations = [inputs]
        # We successively apply each weight matrix to the example using the dot product, and then store that as the
        # activation that should be dotted with the next weight matrix, and so on.
        for i in range(len(self.weights)):
            # We now update input to be the activation of the next layer.
            inputs = np.dot(self.weights[i], inputs)
            # We do not want to apply the sigmoid function if this is the output layer of a regression problem.
            if not (self.is_regression() and i == len(self.weights) - 1):
                inputs = af.sigmoid(inputs)
            # We also do not want to append the activation of the bias node (sigmoid(1)) to the activation if this is
            # the output layer.
            if i < len(self.weights) - 1:
                inputs = np.append(inputs, af.sigmoid(1))
            # We then add the current activation to the list of activations.
            activations.append(inputs)
        # Activation is now a list of numpy arrays, each an activation of a layer.
        return activations

    # Feeds the given observation through the network, returning the output layer's activation.
    # * observation: a numpy array representing the attribute values of a single observation.
    # Returns a numpy array representing the activation of the output layer.
    def feed_forward(self, observation):
        return self.get_activation(observation)[-1]

    # Classifies/regresses a given observation by feeding it forward through the network.
    # * observation: a numpy array representing the attribute values of a single observation.
    # Returns either (1) the class name of the most probable class for the input observation if a classification
    # problem, or (2) the estimated value if a regression problem.
    def run(self, observation):
        output = list(self.feed_forward(observation))
        if self.is_regression():
            return output[0]
        max_class_index = output.index(max(output))
        return self.classes[max_class_index]

    # Returns the root mean squared error on the specified data set according to the current configuration of weights
    # in the network.
    def get_error(self, data_set: DataSet):
        squared_sum = 0
        # Sum up the squared sup across all squared differences between the actual class value and the expected value.
        for example_array, expected_class in data_set.get_data():
            output = self.run(example_array)
            squared_sum += (output - expected_class) ** 2
        return math.sqrt(squared_sum) / len(data_set.get_data())

    # Returns the accuracy on the specified data set according to the current configuration of weights in the network.
    def get_accuracy(self, data_set: DataSet):
        correct = 0
        # Sum the number of correctly classified examples.
        for example_array, expected_class in data_set.get_data():
            output = self.run(example_array)
            if output == expected_class:
                correct += 1
        # Divide the number of correct examples by the total number of examples.
        return correct / len(data_set.get_data())


# Generates a matrix with random weights generated according to a normal distribution centered at zero with a standard
# deviation of one.
def random_weight_matrix(num_in, num_out):
    return np.random.randn(num_out, num_in)
