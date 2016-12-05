from itertools import combinations, product
from discretizers import Discretizers

def split_dataset(dataset):
    transposed = Discretizers.transpose(dataset)
    classes = transposed[-1]
    unique_classes = set(classes)
    transposed = transposed[:-1]
    discretized_dataset = []
    for column in transposed:
        discretized_dataset.append(Discretizers.equal_width_discretizer(column, 3)[0])
    discretized_dataset.append(classes)
    discretized_dataset = Discretizers.transpose(discretized_dataset)
    separated = {klass: [] for klass in classes}
    for record in discretized_dataset:
        klass = record[-1]
        separated[klass].append(record[:-1])
    return separated
