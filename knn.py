#!/usr/bin/env python3
from collections import Counter
from data_loader import DataLoader
import math
import ipdb

def euclidean_distance(x, y):
    return math.sqrt(sum([math.pow((a - b), 2) for a, b in zip(x, y)]))

def manhattan_distance(x, y):
    return sum(abs([(a - b) for a, b in zip(x, y)]))

def plurality_voting(nearest_neighbours):
    classes = [nearest_neighbour[1] for nearest_neighbour in nearest_neighbours]
    count = Counter(classes)
    return count.most_common()[0][0]

def weighted_distance_voting(nearest_neighbours):
    distances = [(1/nearest_neighbour[0], nearest_neighbour[1]) for nearest_neighbour in nearest_neighbours]
    index = distances.index(min(distances))
    return nearest_neighbours[index][1]

def weighted_distance_squared_voting(nearest_neighbours):
    distances = list(map(lambda x: 1 / x[0]*x[0], nearest_neighbours))
    index = distances.index(min(distances))
    return nearest_neighbours[index][1]

def get_neighbours(training_set, test_instance, k):
    names = [instance[4] for instance in training_set]
    training_set = [instance[0:4] for instance in training_set]
    distances = [euclidean_distance(test_instance, training_set_instance) for training_set_instance in training_set]
    distances = list(zip(distances, names))
    sorted(distances, key=lambda x: x[0])
    return distances[:k]

def main():
    data = DataLoader.load_arff("datasets/iris.arff")
    dataset = data["data"]
    train = dataset[:100]
    nn = get_neighbours(train, dataset[120][0:4], 4)
    weighted_distance_voting(nn)
    ipdb.set_trace()

if __name__ == "__main__":
    main()


