# backpropagation.py
# Defines the back propagation algorithm for learning a neural network.

import math
import numpy as np
import src.util.activations as af
from src.network import Network
from src.data import DataSet

# The amount each new average metric needs to be better than the old average metric for the training process to
# continue.
CONVERGENCE_THRESHOLD = .0001


# Used to create an instance of the back propagation algorithm. Trains a neural network according to a set of (tunable)
# parameters.
class BackPropagation:

    # Creates a new BackPropagation object.
    # * network: the network object to train.
    # * batch_size: the size of each mini batch.
    # * learning_rate: the learning rate for the backpropagation algorithm.
    # * momentum: the momentum to carry forward between batches.
    # * convergence_size: the sample size of previous epochs to examine when determining convergence.
    def __init__(self, network: Network, batch_size: int, learning_rate: float, momentum: float, converge_size: int):
        self.network = network
        self.mini_batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.convergence_size = converge_size

    # Trains the network on the training data, ending when the accuracy/error has converged on the validation data.
    def train(self):
        training_data, validation_data = self.network.training_data.partition(.8)
        convergence_check = []

        while True:
            # Check for convergence by evaluating the past self.convergence_size*2 validation metrics (either accuracy
            # or error). We exit if the older half of metrics has a better average than the newer half.
            metric = self.get_error(validation_data) if self.network.is_regression() else \
                self.get_accuracy(validation_data)
            print(metric)
            convergence_check.append(metric)
            # Wait until the convergence check list has all self.convergence_size*2 items.
            if len(convergence_check) > self.convergence_size * 2:
                # Remove the oldest metric, to maintain the list's size.
                convergence_check.pop(0)
                # The last half of the list are the older metrics.
                old_metric = sum(convergence_check[:self.convergence_size])
                # The first half of the list are the newer metrics.
                new_metric = sum(convergence_check[self.convergence_size:])
                # We compare the difference in sums. We could use averages, but there is no difference when comparing
                # the sums or averages since the denominator would be the same size for both.
                difference = new_metric - old_metric
                if self.network.is_regression():
                    # Error needs to invert the difference, as we are MINIMIZING error.
                    if -difference < CONVERGENCE_THRESHOLD:
                        return
                else:
                    # We attempt to MAXIMIZE accuracy for classification data.
                    if difference < CONVERGENCE_THRESHOLD:
                        return

            # If we are here, then there was no convergence. We therefore need to train on the training data (again). We
            # first shuffle the training data so that we aren't learning on the exact same mini batches as last time.
            training_data.shuffle()
            # Now we form the mini batches. Each mini batch is a list of examples.
            mini_batches = [training_data.get_data()[k:k + self.mini_batch_size] for k in
                            range(0, len(training_data.get_data()), self.mini_batch_size)]
            # We now perform gradient descent on each mini batch. We also maintain the delta weights from the previous
            # mini batch so that we can apply momentum to our current delta weight.
            prev_delta_weights = None
            for mini_batch in mini_batches:
                prev_delta_weights = self.train_mini_batch(mini_batch, prev_delta_weights)

    # weights from the last mini batch, or None if this is the first mini batch.
    def train_mini_batch(self, mini_batch: list, prev_weights: list):
        total_dw = [np.zeros(w.shape) for w in self.network.weights]
        for example_array, expected_class in mini_batch:
            expected_array = self.network.expected_output_vector(expected_class)
            delta_weights = self.back_propagation(example_array, expected_array)
            for i in range(len(self.network.weights)):
                delta_weights[i] *= (self.learning_rate / len(mini_batch))
                if prev_weights is not None:
                    delta_weights[i] -= self.momentum * prev_weights[i]
                self.network.weights[i] -= delta_weights[i]
                total_dw[i] += delta_weights[i]
        return total_dw

    # Returns the root mean squared error on the specified data set according to the current configuration of weights
    # in the network.
    def get_error(self, data_set: DataSet):
        squared_sum = 0
        # Sum up the squared sup across all squared differences between the actual class value and the expected value.
        for example_array, expected_class in data_set.get_data():
            output = self.network.run(example_array)
            squared_sum += (output - expected_class) ** 2
        return math.sqrt(squared_sum) / len(data_set.get_data())

    # Returns the accuracy on the specified data set according to the current configuration of weights in the network.
    def get_accuracy(self, data_set: DataSet):
        correct = 0
        # Sum the number of correctly classified examples.
        for example_array, expected_class in data_set.get_data():
            output = self.network.run(example_array)
            if output == expected_class:
                correct += 1
        # Divide the number of correct examples by the total number of examples.
        return correct / len(data_set.get_data())

    # Performs back propagation of a specified example given an expected output.
    def back_propagation(self, example: np.ndarray, expected: np.ndarray):
        # Objective: calculate the change in weights (delta_weights) for the gradient formed by this training example.
        delta_weights = [np.zeros(w.shape) for w in self.network.weights]
        n = len(self.network.layer_sizes)

        # First we perform feedforward, and save all the activations of each layer.
        activation = self.network.get_activation(example)

        # We then calculate the delta of the output layer.
        delta = af.cost_prime(expected, activation[-1])
        if not self.network.is_regression():
            delta = delta * af.sigmoid_prime(activation[-1])
        # The change in weights for the output layer is then updated using delta.
        delta_weights[-1] = np.outer(delta, activation[-2])

        # Now, we calculate the delta and change in weights for each hidden layer weights. The variable i represents the
        # layer we are currently evaluating, with i=1 being the first hidden layer, i=2 being the second, etc.
        for i in range(n - 2, 0, -1):
            # This is the activation from the previous hidden (or the input) layer.
            previous_activation = activation[i - 1]
            # These are the weights going out from the current layer.
            downstream_weights = self.network.weights[i].T

            # We need to trim the delta from the previous layer (unless it comes from the output layer) because it
            # includes a delta for the bias node, which does not change.
            if i < n - 2:
                delta = delta[:-1]
            # We compute delta for this layer:
            delta = np.dot(downstream_weights, delta) * af.sigmoid_prime(activation[i])
            # Then we update the delta_weight matrix to have the change in weights for this layer's weights:
            delta_weights[i - 1] = np.delete(np.outer(delta, previous_activation), -1, 0)
        # We return delta_weights, which is a list of the matrices to be subtracted from the actual weights.
        return delta_weights
