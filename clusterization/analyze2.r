library(clusterCrit)
library(cluster)
library(foreign)
library(clue)

datasets = c("wholesale.csv")

append.results <- function(summary, filename) {
  write.table(
    summary,
    filename,
    append = TRUE,
    quote = FALSE,
    sep = ",",
    row.names = FALSE,
    col.names = FALSE
  )
}

clusterize <- function(algorithm = kmeans,
                       clusters = 2,
                       data) {
  result <- algorithm(data, clusters)
  data[] <- lapply(data, as.numeric)
  summary = intCriteria(as.matrix(data),
                        result$cluster,
                        c("Dunn", "Silhouette", "Davies_Bouldin"))
  return(list(clusters = result, indices = summary))
}

crossValidate <-
  function(algorithm,
           no_of_clusters,
           no_of_folds,
           dataset) {
    fun <- match.fun(algorithm)
    dataframe <- read.csv(sprintf("datasets/%s", dataset))
    dataframe <- dataframe[sample(nrow(dataframe)), ]
    folds <- cut(seq(1, nrow(dataframe)), breaks = folds, labels = FALSE)
    results <- data.frame()
    for (i in 1:no_of_folds) {
      testIndexes <- which(folds == i, arr.ind = TRUE)
      testData <- dataframe[, -ncol(dataframe)][testIndexes,]
      
      trainData <- dataframe[, -ncol(dataframe)][-testIndexes,]
      
      cluserization <-
        clusterize(fun, no_of_clusters, data = trainData)
      
      result <- list(clusters = clusters, dunn = cluserization$indices$dunn, silhouette = cluserization$indices$silhouette, davies_bouldin = cluserization$indices$davies_bouldin)
      results <- rbind(results, result)
    }
    browser()
    means <- colMeans(results)
    wtf <- as.matrix(t(means))

    append.results(wtf,
                   sprintf("tmp/%s_%s_%s.csv", algorithm, dataset, no_of_folds))
    return(results)
  }

for (algorithm in c("kmeans", "pam")) {
  for (dataset in datasets) {
    for (folds in c(3, 5, 10)) {
      for (clusters in 2:10) {
        crossValidate(algorithm, clusters, folds, dataset)
      }
    }
  }
}
