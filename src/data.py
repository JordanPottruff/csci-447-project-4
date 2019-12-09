# data.py
# Defines the data components of the project.

import math
import random
import csv
import numpy as np


# Used to represent a particular data set. The underlying representation of the data is discussed in the comments for
# the constructor and get_data methods. A data set object can be used for a variety of tasks, including normalization
# of attributes, shuffling of observations, and creation of folds for cross validation.
class DataSet:

    # Creates a new DataSet object.
    # * data: a list of observations represented as tuples where the first position in the tuple is the numpy float
    #   array of the attribute columns, and the second position is the class value (either a string in the case of
    #   classification or float if regression).
    def __init__(self, data: list):
        self.data = data
        self.num_cols = len(data[0][0])

    # Returns a string representing the underlying data in a tabular format, where each row is an observation in the
    # data set.
    def __str__(self):
        str_rep = ""
        for observation in self.data:
            str_rep += "({}, {})".format(observation[0].tolist(), observation[1]) + "\n"
        return str_rep

    # Returns the underlying data, formatted as the list of tuples described in the constructor.
    def get_data(self):
        return self.data

    # Normalizes all attribute columns of the data set, in place. Normalization is performed with the assumption that
    # the attribute values are roughly distributed according to a normal distribution, and therefore can be replaced
    # with their corresponding z-scores. These z-scores represent how many standard deviations the particular
    # observation's attribute value was from the mean of that attribute.
    def normalize(self):
        for i in range(self.num_cols):
            # For each attribute, get the sum of the values and calculate the mean...
            attr_sum = 0
            for row in self.data:
                attr_sum += row[0][i]
            attr_mean = attr_sum / len(self.data)

            # ...and then use the mean to find the standard deviation...
            attr_squared_diff_sum = 0
            for row in self.data:
                attr_squared_diff_sum += (attr_mean - row[0][i])**2
            attr_standard_dev = math.sqrt(attr_squared_diff_sum)

            # ...and then use the mean and standard deviation to compute z_scores to replace attribute values with.
            for row in self.data:
                z_score = (row[0][i] - attr_mean) / attr_standard_dev if attr_standard_dev != 0 else 0
                row[0][i] = z_score

    # Shuffles the observations into a new random order, in place.
    def shuffle(self):
        random.shuffle(self.data)

    # Partitions the data set into two new data sets that split the underlying observations of the current one. This
    # partitioning does not affect the current data set, but rather returns two entirely new data sets that share the
    # same observations.
    # * first_percentage: the proportion of observations that should fall in the first data set, with the remainder
    #   falling in the second one.
    # Returns the two data sets as a tuple.
    def partition(self, first_percentage: float):
        cutoff = math.floor(first_percentage * len(self.data))
        first = DataSet(self.data[:cutoff])
        second = DataSet(self.data[cutoff:])
        return first, second

    # Creates n number of 'folds' from the current data, for use with cross validation. A fold is made up of two data
    # sets: a training set,  and a testing set. The test set contains 1/n observations, while the training set is
    # composed of the remaining n-1/n observations. The DataSet objects created for each fold are newly initialized, and
    # the process of creating these folds does not affect the current data set.
    # * n: the number of folds to generate.
    # Returns a list of n folds, where each fold is a dictionary. The testing set can be accessed via the 'test' key,
    # and the training set via the 'train' key.
    def validation_folds(self, n: int):
        avg_size = math.ceil(len(self.data) / n)
        # Create n segments of a (roughly) equal number of observations.
        segments = [self.data[round(k * avg_size): round((k + 1) * avg_size)] for k in range(10)]

        # Iterate through, making each segment a test set of a fold and the remaining segments the training set.
        folds = [{} for i in range(n)]
        for i in range(n):
            # Each test and training component are a DataSet, not a direct list of numpy arrays.
            folds[i]['test'] = DataSet(segments[i])
            training_data = []
            for segment_i in range(n):
                for elem in segments[segment_i]:
                    if segment_i != i:
                        training_data.append(elem)
            folds[i]['train'] = DataSet(training_data)
        return folds


#
# Below are functions for creating DataSet objects of the various data files as well as helper code.
#


# Returns the abalone data as a DataSet object.
def get_abalone_data(file_name="../data/abalone.data", normalize=True):
    data = read_file(file_name)
    data = format_data(data, [1, 2, 3, 4, 5, 6, 7], 8)

    abalone_data = DataSet(data)
    if normalize:
        abalone_data.normalize()
    abalone_data.shuffle()
    return abalone_data


# Returns the car data as a DataSet object.
def get_car_data(file_name="../data/car.data", normalize=True):
    data = read_file(file_name)
    convert_attribute(data, 0, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    convert_attribute(data, 1, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    convert_attribute(data, 2, {'2': 2, '3': 3, '4': 4, '5more': 5})
    convert_attribute(data, 3, {'2': 2, '4': 4, 'more': 5})
    convert_attribute(data, 4, {'small': 0, 'med': 1, 'big': 2})
    convert_attribute(data, 5, {'low': 0, 'med': 1, 'high': 2})
    data = format_data(data, [0, 1, 2, 3, 4, 5], 6)

    car_data = DataSet(data)
    if normalize:
        car_data.normalize()
    car_data.shuffle()
    return car_data


# Returns the forest fire data as a DataSet object.
def get_forest_fire_data(file_name="../data/forestfires.data", normalize=True):
    data = read_file(file_name, 1)
    convert_attribute(data, 2, {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
    convert_attribute(data, 3, {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7})
    data = format_data(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 12)

    forest_fire_data = DataSet(data)
    if normalize:
        forest_fire_data.normalize()
    forest_fire_data.shuffle()
    return forest_fire_data


# Returns the machine data as a a DataSet object.
def get_machine_data(file_name="../data/machine.data", normalize=True):
    data = read_file(file_name)
    data = format_data(data, [2, 3, 4, 5, 6, 7], 8)

    machine_data = DataSet(data)
    if normalize:
        machine_data.normalize()
    machine_data.shuffle()
    return machine_data


# Returns the segmentation data as a DataSet object.
def get_segmentation_data(file_name="../data/segmentation.data", normalize=True):
    data = read_file(file_name, 5)
    data = format_data(data, list(range(1, 20)), 0)

    segmentation_data = DataSet(data)
    if normalize:
        segmentation_data.normalize()
    segmentation_data.shuffle()
    return segmentation_data


# Returns the wine data as a DataSet object.
def get_wine_data(file_name="../data/winequality.data", normalize=True):
    data = read_file(file_name)
    data = format_data(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)

    wine_data = DataSet(data)
    if normalize:
        wine_data.normalize()
    wine_data.shuffle()
    return wine_data


# Converts the given value to a float, if possible. If not, returns the original value.
def floatify(value):
    try:
        float(value)
    except ValueError:
        return value
    return float(value)


# Creates a 2D list from a file.
def read_file(filename: str, header_size=0):
    with open(filename) as csvfile:
        data = list(csv.reader(csvfile))
    cleaned_data = []
    for line in data[header_size:]:
        if line:
            cleaned_data.append(line)
    return cleaned_data


# Convert specified attribute (attr_col) according to the provided conversion map.
def convert_attribute(data: list, attr_col: int, conversion_map: dict):
    for row in data:
        if row[attr_col] in conversion_map:
            row[attr_col] = conversion_map[row[attr_col]]


# Formats data as a list of tuples that represent each observation. The first position in the tuple is the numpy float
# array of the attribute columns, and the second position is the class value (either a string in the case of
# classification or float if regression).
# * data: the list of examples, with each one either being a list, tuple, or numpy array.
# * attr_cols: list of indices for the attribute columns.
# * class_col: the index of the class column.
# Returns the list of data, formatted as a tuple in the way described above.
def format_data(data: list, attr_cols: list, class_col: int):
    return [(np.array([float(obs[i]) for i in attr_cols]), floatify(obs[class_col])) for obs in data]
