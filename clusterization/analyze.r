library(clusterCrit)
library(cluster)
library(foreign)
library(clue)

accuracy <-
  function(confusion_matrix,
           real_classes,
           predicted_classes) {
    tot = sum(confusion_matrix)
    if (tot == 0) {
      return(tot)
    }
    tptn <- 0
    for (real_class in real_classes) {
      i <- sum(which(real_classes == real_class))
      j <- sum(which(predicted_classes == real_class))
      f <- 0
      if (j != 0) {
        f <- confusion_matrix[i, j]
      }
      tptn <- tptn + f
    }
    return(tptn / tot)
  }

recall <- function(confusion_matrix,
                   real_classes,
                   predicted_classes) {
  recall_per_class <- c()
  for (real_class in real_classes) {
    e <- 0
    for (predicted_class in predicted_classes) {
      i <- which(real_classes == real_class)
      j <- which(predicted_classes == predicted_class)
      e <- e + confusion_matrix[i, j]
    }
    if (e != 0) {
      i <- sum(which(real_classes == real_class))
      j <- sum(which(predicted_classes == real_class))
      f <- 0
      if (j != 0) {
        f <- confusion_matrix[i, j]
      }
      e <- f / e
    }
    recall_per_class <- c(recall_per_class, e)
  }
  return(sum(recall_per_class) / length(recall_per_class))
}

precission <-
  function(confusion_matrix,
           real_classes,
           predicted_classes) {
    prediction_per_class <- c()
    for (predicted_class in predicted_classes) {
      e <- 0
      for (real_class in real_classes) {
        i <- which(real_classes == real_class)
        j <- which(predicted_classes == predicted_class)
        e <- e + confusion_matrix[i, j]
      }
      if (e != 0) {
        i <- sum(which(real_classes == predicted_class))
        j <- sum(which(predicted_classes == predicted_class))
        f <- 0
        if (j != 0) {
          f <- confusion_matrix[i, j]
        }
        e <- f / e
      }
      prediction_per_class <- c(prediction_per_class, e)
    }
    return(sum(prediction_per_class) / length(prediction_per_class))
  }

fscore  <-
  function(confusion_matrix,
           real_classes,
           predicted_classes) {
    e <-
      recall(confusion_matrix, real_classes, predicted_classes) + precission(confusion_matrix, real_classes, predicted_classes)
    if (e == 0) {
      return(e)
    }
    return((
      2 * precission(confusion_matrix, real_classes, predicted_classes) * recall(confusion_matrix, real_classes, predicted_classes)
    ) / e)
  }

assign_to_cluster <- function(x, centers) {
  tmp <-
    sapply(seq_len(nrow(x)), function(i)
      apply(centers, 1, function(v)
        sum((x[i,] - v) ^ 2)))
  max.col(-t(tmp))  # find index of min distance
}

get_cluster_class <- function(cluster_class, cluster) {
  classes_in_cluster <- as.matrix(cluster_class)[cluster, ]
  freq <- table(classes_in_cluster)
  class = names(which.max(freq))[1]
  return(class)
}

purity <- function(cluster_class, classes) {
  purity_val <-
    length(which(classes == cluster_class)) / length(classes)
  return(purity_val)
}

datasets = c("iris.arff", "diabetes.arff", "ionosphere.arff")

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
    dataframe <- read.arff(sprintf("datasets/%s", dataset))
    dataframe <- dataframe[sample(nrow(dataframe)), ]
    folds <- cut(seq(1, nrow(dataframe)), breaks = folds, labels = FALSE)
    results <- data.frame()
    confusion_matrix = NULL
    for (i in 1:no_of_folds) {
      testIndexes <- which(folds == i, arr.ind = TRUE)
      testData <- dataframe[, -ncol(dataframe)][testIndexes,]
      testDataColumns <- dataframe[testIndexes, ncol(dataframe)]
      
      trainData <- dataframe[, -ncol(dataframe)][-testIndexes,]
      trainDataColumns <- dataframe[-testIndexes, ncol(dataframe)]
      
      cluserization <-
        clusterize(fun, no_of_clusters, data = trainData)
      
      if(algorithm == "kmeans") {
        assigned_clusters <-
          assign_to_cluster(testData, cluserization$clusters$centers)
      }
      else {
        assigned_clusters <-
          assign_to_cluster(testData, cluserization$clusters$medoids)
      }
      
      predicted_classes <- assigned_clusters
      purity_vals <- c()
      
      for (j in 1:no_of_clusters) {
        cluster_idx = which(assigned_clusters == j)
        cluster_class <-
          get_cluster_class(testDataColumns, cluster_idx)
        
        purity_val <-
          purity(cluster_class, as.matrix(testDataColumns)[cluster_idx, ])
        predicted_classes <-
          replace(predicted_classes, predicted_classes == j, cluster_class)
        purity_vals <- c(purity_vals, purity_val)
      }
      
      confusion_matrix <-
        table(testDataColumns, predicted_classes)
      real_classes <- names(table(testDataColumns))
      predicted_classes <- names(table(predicted_classes))
      
      acc <-
        accuracy(confusion_matrix, real_classes, predicted_classes)
      rec <-
        recall(confusion_matrix, real_classes, predicted_classes)
      prec <-
        precission(confusion_matrix, real_classes, predicted_classes)
      fsc <-
        fscore(confusion_matrix, real_classes, predicted_classes)
      prt <- sum(purity_vals) / length(purity_vals)
      
      #result <- list(clusters = clusters, dunn = cluserization$indices$dunn, silhouette = cluserization$indices$silhouette, davies_bouldin = cluserization$indices$davies_bouldin, accuracy = acc, recall = rec, precision = prec, fscore=fsc, purity = prt)
      results <- rbind(results, result)
    }
    
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
