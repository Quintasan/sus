#!/usr/bin/env python3
from collections import Counter
from data_loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support as pr
import random
import math
import ipdb

def euclidean_distance(x, y):
    return math.sqrt(sum([math.pow((a - b), 2) for a, b in zip(x, y)]))

def manhattan_distance(x, y):
    return sum(abs([(a - b) for a, b in zip(x, y)]))

def get_neighbours(training_set, test_instance, k):
    names = [instance[4] for instance in training_set]
    training_set = [instance[0:4] for instance in training_set]
    distances = [euclidean_distance(test_instance, training_set_instance) for training_set_instance in training_set]
    distances = list(zip(distances, names))
    print(list(filter(lambda x: x[0] == 0.0, distances)))
    sorted(distances, key=lambda x: x[0])
    return distances[:k]

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

def main():
    data = DataLoader.load_arff("datasets/iris.arff")
    dataset = data["data"]
    # random.seed(42)
    random.shuffle(dataset)
    train = dataset[:100]
    test = dataset[100:150]
    classes = [instance[4] for instance in test]
    predictions = []
    for test_instance in test:
        prediction = weighted_distance_voting(get_neighbours(train, test_instance[0:4], 15))
        predictions.append(prediction)
    print(pr(classes, predictions, average="micro"))

if __name__ == "__main__":
    main()
