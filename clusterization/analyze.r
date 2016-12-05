library(cluster)
library(foreign)

datasets = c("datasets/iris.arff", "datasets/diabetes.arff", "datasets/ionosphere.arff")
for (dataset in datasets) {
        data = read.arff(dataset)
        no_of_classes = length(t(unique(data[ncol(data)])))
        numerical = data[,-ncol(data)]
        message(sprintf("K-means clusterer on %s with %s clusters", dataset, no_of_classes))
        print(kmeans(numerical, no_of_classes))
        message(sprintf("PAM clusterer on %s with %s clusters", dataset, no_of_classes))
        print(pam(numerical, no_of_classes))
}
