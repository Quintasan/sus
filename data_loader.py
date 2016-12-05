import arff
import random

class DataLoader:
    @staticmethod
    def load_arff(filename):
        return arff.load(open(filename, "r"))

    @staticmethod
    def split_dataset(dataset, folds):
        random.shuffle(dataset)
        return [dataset[i::folds] for i in range(folds)]
