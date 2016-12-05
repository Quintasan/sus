# https://github.com/fracpete/python-weka-wrapper3/blob/29eb47f1ee59a734b5b019d08852dcbeee187bfe/doc/source/api.rst
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.filters import Filter
from weka.classifiers import FilteredClassifier

def run_weka(filename, options, folds):
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(filename)
    data.class_is_last()

    classifier = Classifier(classname="weka.classifiers.trees.J48")
    classifier.options = options

    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(classifier, data, folds, Random(666))

    s = ",".join(map(lambda x: str(x), [filename, folds, options,
        evaluation.weighted_precision, evaluation.percent_correct,
        evaluation.weighted_recall, evaluation.weighted_f_measure]))
    print(s)

def options_dict_to_list(dd):
    l = []
    for k, v in dd.items():
        l.append(k)
        if v is not None:
            l.append(v)
    return l

if __name__ == '__main__':
    jvm.start()

    datasets = ["datasets/diabetes.arff", "datasets/glass.arff", "datasets/ionosphere.arff"]

    for folds in [3, 5, 10]:
        for dataset in datasets:
            options = {"-C":"0.25","-M":"2"}
            print("==========CONFIDENCE FACTOR WITH {} FOLDS==========".format(folds))
            for confidence_factor in range(5, 55, 5):
                options["-C"] = str(confidence_factor/100)
                run_weka(dataset, options_dict_to_list(options), folds)

            options = {"-C":"0.25","-M":"2"}
            print("==========INSTANCES PER LEAF WITH {} FOLDS==========".format(folds))
            for instances_per_leaf in range(1, 50, 1):
                options["-M"] = str(instances_per_leaf)
                run_weka(dataset, options_dict_to_list(options), folds)

    jvm.stop()
