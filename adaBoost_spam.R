
#############################################################################
# adaBoost_spam.R
#
# Matthew Sudmann-Day
# Barcelona GSE Data Science
#
# Tests the AdaBoost.M1 boosting algorithm implemented in adaBoost.R and
# compares it to the boosting performed by the 'gbm' package.
#
# Uses R packages:
#   assertthat, rpart, gbm, caTools, ggplot2
#
# Public function adaBoost_spam() executes the test and outputs a plot
#   that compares training and test errors for AdaBoost.M1 and GBM.  The plot
#   is returned directly, and is also written to PDF and PNG files.
#
# Parameters:
#   formula: an R-style formula to describe the label and the independent
#     columns in the data
#   data: a data frame containing the training data
#   depth: the maximum depth of the classification true for rpart to create
#   maxTrees: the maximum number of rpart models to create, ie, the maximum
#     number of boosting iterations to test against
#   trainRatio: the ratio of data to use for training, the remainder for test
#
# Returns:
#   a plot of training and test errors for AdaBoost.M1 and GBM
#############################################################################
adaBoost_spam <- function(formula, data, depth, maxTrees, trainRatio)
{
  if (!require('assertthat')) install.packages('assertthat')
  library(assertthat)
  if (!require('rpart')) install.packages('rpart')
  library(rpart)
  if (!require('caTools')) install.packages('caTools')
  library(caTools)
  if (!require('gbm')) install.packages('gbm')
  library(gbm)
  if (!require('ggplot2')) install.packages('ggplot2')
  library(ggplot2)

  # Basic assertions that the parameters are valid.
  assert_that(class(formula) == "formula")
  assert_that(class(data) == "data.frame" && nrow(data) > 0 && ncol(data) > 1)
  assert_that(is.numeric(depth) && depth > 1)
  assert_that(is.numeric(maxTrees) && maxTrees > 0)
  assert_that(is.numeric(trainRatio) && trainRatio > 0 && trainRatio < 1)
  
  # Generate a model frame object to sort out the environment of the formula and
  # to allow us to add a weights column to the data without changing the meaning
  # of the formula in case the formula contains a '.' for all other variables.
  mf <- model.frame(formula, data)
  labels <- mf[,1]
  
  # Split the training and test data.
  spl <- sample.split(data, trainRatio)
  trainData <- data[spl,]
  trainLabels <- labels[spl]
  testData <- data[!spl,]
  testLabels <- labels[!spl]
  
  results <- NULL
  
  # Loop through all the tree counts.
  for (noTrees in 1:maxTrees)
  {
    cat(paste("  noTrees=", noTrees, "\n", sep=""))
    
    # Measure AdaBoost.M1 training errors.
    preds <- adaBoost(formula, trainData, depth, noTrees)
    errors <- mean(preds$predLabels != trainLabels)
    results <- rbind(results, data.frame(noTrees=noTrees, errors=errors, Error_Type="AdaBoost.M1 Train"))
    
    # Measure AdaBoost.M1 test errors.
    preds <- adaBoost(formula, trainData, depth, noTrees, testData)
    errors <- mean(preds$predLabels != testLabels)
    results <- rbind(results, data.frame(noTrees=noTrees, errors=errors, Error_Type="AdaBoost.M1 Test"))
    
    # Measure GBM training errors.
    mdl <- gbm(formula=formula, distribution="adaboost", data=trainData, n.trees=noTrees,
               interaction.depth=depth, shrinkage=1, bag.fraction=1)
    preds <- predict(mdl, newdata=trainData, n.trees=noTrees)
    errors <- mean(ifelse(preds > 0, 1, 0) != trainLabels)
    results <- rbind(results, data.frame(noTrees=noTrees, errors=errors, Error_Type="GBM Train"))
    
    # Measure GBM test errors.
    mdl <- gbm(formula=formula, distribution="adaboost", data=trainData, n.trees=noTrees,
               interaction.depth=depth, shrinkage=1, bag.fraction=1)
    preds <- predict(mdl, newdata=testData, n.trees=noTrees)
    errors <- mean(ifelse(preds > 0, 1, 0) != testLabels)
    results <- rbind(results, data.frame(noTrees=noTrees, errors=errors, Error_Type="GBM Test"))
  }
  
  # Produce a plot, export it, and return it.
  plot <- ggplot(results, aes(x=noTrees, y=errors)) + geom_line(aes(color=Error_Type))
  ggsave("adaBoost.pdf", plot)
  ggsave("adaBoost.png", plot)
  
  return(plot)
}

# Load the spam data set and initiate the test.
data <- read.csv("Spam/spambase.data")
adaBoost_spam(spam ~ ., data, 8, 60, 0.7)
