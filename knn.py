#!/usr/bin/env python3
from collections import Counter
from data_loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support as pr
import random
import math
import ipdb
import csv

def k_fold_cross_validation(X, K, randomise = False):
    if randomise: X=list(X); random.shuffle(X)
    for k in range(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

def euclidean_distance(x, y):
    return math.sqrt(sum([math.pow((a - b), 2) for a, b in zip(x, y)]))

def manhattan_distance(x, y):
    return sum([abs(a - b) for a, b in zip(x, y)])

def get_neighbours(name, training_set, test_instance, k):
    if name == "wine.arff":
        names = [instance[0] for instance in training_set]
        training_set = [instance[1:] for instance in training_set]
    else:
        names = [instance[-1] for instance in training_set]
        training_set = [instance[:-1] for instance in training_set]
    distances = [manhattan_distance(test_instance, training_set_instance) for training_set_instance in training_set]
    distances = list(zip(distances, names))
    distances = sorted(distances, key=lambda x: x[0])
    return distances[:k]

def plurality_voting(nearest_neighbours):
    classes = [nearest_neighbour[1] for nearest_neighbour in nearest_neighbours]
    count = Counter(classes)
    return count.most_common()[0][0]

def weight_distance(neighbour):
    try:
        distance = 1/neighbour[0]
    except ZeroDivisionError:
        distance = float("inf")
    return (distance, neighbour[1])

def weighted_distance_voting(nearest_neighbours):
    distances = [weight_distance(nearest_neighbour) for nearest_neighbour in nearest_neighbours]
    index = distances.index(min(distances))
    return nearest_neighbours[index][1]

def weight_squared_distance(neighbour):
    try:
        distance = 1/(neighbour[0] * neighbour[0])
    except ZeroDivisionError:
        distance = float("inf")
    return (distance, neighbour[1])

def weighted_distance_squared_voting(nearest_neighbours):
    distances = [weight_squared_distance(nearest_neighbour) for nearest_neighbour in nearest_neighbours]
    index = distances.index(min(distances))
    return nearest_neighbours[index][1]

def main():
    filenames = ["iris.arff", "glass.arff", "wine.arff", "diabetes.arff", "ionosphere.arff"]
    for filename in filenames:
        dataset = DataLoader.load_arff("datasets/{0}".format(filename))["data"]
        results = []
        print(filename)
        for folds in [3, 5, 10]:
            print("{0}-fold crossvalidation".format(folds))
            for param_k in [1,2,3,4,5,6,7,8,9,10]:
                print("k = {0}".format(param_k))
                for training, validation in k_fold_cross_validation(dataset, K=folds, randomise = True):
                    if filename == "wine.arff":
                        classes = [instance[0] for instance in validation]
                    else:
                        classes = [instance[-1] for instance in validation]
                    predictions = []
                    average_method = "macro"
                    for instance in validation:
                        if filename == "wine.arff":
                            nearest_neighbours = get_neighbours(filename, training, instance[1:], param_k)
                        else:
                            nearest_neighbours = get_neighbours(filename, training, instance[:-1], param_k)
                        prediction = weighted_distance_squared_voting(nearest_neighbours)
                        predictions.append(prediction)
                    results.append(pr(classes, predictions, average=average_method)[:-1])
                average = list(map(lambda x: sum(x)/len(x), zip(*results)))
                with open("tmp/manhattan_squared_{0}_{1}.csv".format(filename, folds), "a", newline="") as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerow(average)

if __name__ == "__main__":
    main()
