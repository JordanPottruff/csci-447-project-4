
import math
import random
import csv
import numpy as np


class DataSet:

    def __init__(self, data, attr_cols, class_col):
        self.data = [(np.array([float(obs[i]) for i in attr_cols]), floatify(obs[class_col])) for obs in data]
        self.attr_cols = attr_cols
        self.class_col = class_col

    def __str__(self):
        str_rep = ""
        for observation in self.data:
            str_rep += "({}, {})".format(observation[0].tolist(), observation[1]) + "\n"
        return str_rep

    def get_data(self):
        return self.data

    def normalize(self):
        for i in range(len(self.attr_cols)):
            attr_sum = 0
            for row in self.data:
                attr_sum += row[0][i]
            attr_mean = attr_sum / len(self.data)

            attr_squared_diff_sum = 0
            for row in self.data:
                attr_squared_diff_sum += (attr_mean - row[0][i])**2
            attr_standard_dev = math.sqrt(attr_squared_diff_sum)

            for row in self.data:
                z_score = (row[0][i] - attr_mean) / attr_standard_dev if attr_standard_dev != 0 else 0
                row[0][i] = z_score

    def shuffle(self):
        random.shuffle(self.data)

    def partition(self, first_percentage):
        cutoff = math.floor(first_percentage * len(self.data))
        first = DataSet(self.data[:cutoff], self.attr_cols, self.class_col)
        second = DataSet(self.data[cutoff:], self.attr_cols, self.class_col)
        return first, second

    def validation_folds(self, n):
        avg_size = math.ceil(len(self.data) / n)
        segments = [self.data[round(k * avg_size): round((k + 1) * avg_size)] for k in range(10)]

        folds = [{} for i in range(n)]
        for i in range(n):
            folds[i]['test'] = DataSet(segments[i], self.attr_cols, self.class_col)
            folds[i]['train'] = DataSet(segments[:i] + segments[i+1:], self.attr_cols, self.class_col)
        return folds


def get_abalone_data(file_name="../data/abalone.data", normalize=True):
    data = read_file(file_name)

    abalone_data = DataSet(data, [1, 2, 3, 4, 5, 6, 7], 8)
    if normalize:
        abalone_data.normalize()
    return abalone_data


def get_car_data(file_name="../data/car.data", normalize=True):
    data = read_file(file_name)
    convert_attribute(data, 0, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    convert_attribute(data, 1, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    convert_attribute(data, 2, {'2': 2, '3': 3, '4': 4, '5more': 5})
    convert_attribute(data, 3, {'2': 2, '4': 4, 'more': 5})
    convert_attribute(data, 4, {'small': 0, 'med': 1, 'big': 2})
    convert_attribute(data, 5, {'low': 0, 'med': 1, 'high': 2})

    car_data = DataSet(data, [0, 1, 2, 3, 4, 5], 6)
    if normalize:
        car_data.normalize()
    return car_data


def get_forest_fire_data(file_name="../data/forestfires.data", normalize=True):
    data = read_file(file_name, 1)
    convert_attribute(data, 2, {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
    convert_attribute(data, 3, {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7})

    forest_fire_data = DataSet(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 12)
    if normalize:
        forest_fire_data.normalize()
    return forest_fire_data


def get_machine_data(file_name="../data/machine.data", normalize=True):
    data = read_file(file_name)

    machine_data = DataSet(data, [2, 3, 4, 5, 6, 7], 8)
    if normalize:
        machine_data.normalize()
    return machine_data


def get_segmentation_data(file_name="../data/segmentation.data", normalize=True):
    data = read_file(file_name, 5)

    segmentation_data = DataSet(data, list(range(1, 20)), 0)
    if normalize:
        segmentation_data.normalize()
    return segmentation_data


def get_wine_data(file_name="../data/winequality.data", normalize=True):
    data = read_file(file_name)

    wine_data = DataSet(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 11)
    if normalize:
        wine_data.normalize()
    return wine_data


# Converts the given value to a float, if possible. If not, returns the original value.
def floatify(value):
    try:
        float(value)
    except ValueError:
        return value
    return float(value)


# Creates a 2D list from a file.
def read_file(filename, header_size=0):
    with open(filename) as csvfile:
        data = list(csv.reader(csvfile))
    cleaned_data = []
    for line in data[header_size:]:
        if line:
            cleaned_data.append(line)
    return cleaned_data


# Convert specified attribute (attr_col) according to the provided conversion map.
def convert_attribute(data, attr_col, conversion_map):
    for row in data:
        if row[attr_col] in conversion_map:
            row[attr_col] = conversion_map[row[attr_col]]
