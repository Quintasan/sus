import math
import itertools
from collections import OrderedDict
import ipdb

class Discretizers:

    @staticmethod
    def transpose(data):
        return list(map(lambda x: list(x),(zip(*data))))

    @staticmethod
    def equal_width_discretizer(column, bins):
        new_column = []
        diff = (max(column) - min(column)) / bins

        edges = []
        for i in range(0, bins):
            edges.append(min(column) + i * diff)
        edges.append(max(column))

        for value in column:
            for i, edge in enumerate(edges[1:]):
                if value <= edge:
                    new_column.append(i)
                    break

        return new_column, edges


    @staticmethod
    def frequency_discretizer(column, bins):
        w = math.ceil(len(column)/bins)
        scol = sorted(column)
        freq = {key: len(list(filter(lambda x: x == key, scol))) for key in set(scol)}
        freq = OrderedDict(sorted(freq.items(), key=lambda x: x[0]))
        new_col = []
        bin = []
        for key, frq in freq.items():
            if not len(bin)+frq <= w:
                new_col.append(bin)
                bin = []
            for i in range(frq):
                bin.append(key)
        new_col.append(bin)
        new_col = filter(lambda x: x != [], new_col)
        replace_dict = {}
        for i, bin in enumerate(new_col):
            for val in set(bin):
                replace_dict[val] ="bin{}".format(i)
        return [replace_dict[i] for i in column]
