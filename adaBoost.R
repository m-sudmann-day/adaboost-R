
#############################################################################
# adaBoost.R
#
# Matthew Sudmann-Day
# Barcelona GSE Data Science
#
# Implements an AdaBoost.M1 boosting algorithm to boost the weak classifier
# rpart.  Provides a 
#
# Uses R packages:
#   assertthat, rpart
#
# Public function adaBoost() uses an AdaBoost.M1 boosting algorithm to boost
# the weak classifier rpart.
#
# Parameters:
#   formula: an R-style formula to describe the label and the independent
#     columns in the data
#   data: a data frame containing the training data
#   depth: the maximum depth of the classification true for rpart to create
#   noTrees: the number of rpart models to create, ie, the number of boosting
#     iterations
#   test: a data frame containing the test data.  If not provided, the
#     training data will also be used as the test data.
#
# Returns:
#   a list with one element:
#     predLabels: predicted labels
#############################################################################
adaBoost <- function(formula, data, depth, noTrees, test=data)
{
  if (!require('assertthat')) install.packages('assertthat')
  library(assertthat)
  if (!require('rpart')) install.packages('rpart')
  library(rpart)


  # Basic assertions that the parameters are valid.
  assert_that(class(formula) == "formula")
  assert_that(class(data) == "data.frame" && nrow(data) > 0 && ncol(data) > 1)
  assert_that(is.numeric(depth) && depth > 1)
  assert_that(is.numeric(noTrees) && noTrees > 0)
  assert_that(class(test) == "data.frame" && nrow(test) > 0 && ncol(test) > 1)
  
  # Generate a model frame object to sort out the environment of the formula and
  # to allow us to add a weights column to the data without changing the meaning
  # of the formula in case the formula contains a '.' for all other variables.
  mf <- model.frame(formula, data)
  formula2 <- as.formula(mf)
  
  # Pull the true labels from the model frame.
  labels <- mf[,1]

  # Get a vector of unique labels.  Assert that it be of length 2.
  uniqueLabels <- unique(levels(factor(labels)))
  assert_that(length(uniqueLabels) == 2)
  
  # Initialize the weights, alpha (our model "scores"), and G (our list of models).
  N <- nrow(data)
  w <- rep(1/N, N)
  alpha <- rep(NA, noTrees)
  G <- list()
  
  # Loop to create the number of trees we need.
  for (m in 1:noTrees)
  {
    # Copy the weights vector into the model frame for use by rpart.
    mf$w<-w

    # Create a classification tree model using rpart.
    G[[m]] <- rpart(formula2, data=mf, weights=w, method="class",
                    control=rpart.control(maxdepth=depth))

    # Apply that model back to the training data to get predictions.
    preds <- predict(G[[m]], newdata=mf, type="class")
    preds <- as.numeric(levels(preds)[preds])
    
    # Generate a vector of misclassifications.
    misclass <- ifelse(preds != labels, 1, 0)

    # Using the Adaboost.M1 algorithm, update the observation weights
    # based on our misclassifications.
    err <- sum(w * misclass)/sum(w)
    alpha[m] <- log((1-err)/err)
    w <- w * exp(alpha[m] * misclass)
  }

  # Initialize a vector of meta predictions.
  metaPreds <- rep(0, nrow(test))

  # Loop through all the trees generating predictions from each model.
  for (m in 1:noTrees)
  {
    # Generate predictions from model m on our test data.
    preds <- predict(G[[m]], newdata=test, type="class")

    # Map these predictions into {-1, 1}.
    posNegPreds <- ifelse(preds == uniqueLabels[1], -1, 1)

    # Update the meta predictions.
    metaPreds <- metaPreds + (alpha[m] * posNegPreds)
  }

  # Map these meta-predictions back from {-1, 1} to their original values.
  metaPreds <- ifelse(metaPreds < 0, uniqueLabels[1], uniqueLabels[2])
  
  # Return our results.
  return(list(predLabels=metaPreds))
}
