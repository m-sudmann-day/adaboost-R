# Implementation of AdaBoost.M1

Implements the AdaBoost.M1 algorithm by boosting a weak classifier from the 'rpart' package and
testing it against the boosting in the 'gbm' package.

For more detail, please see the comments in the R scripts.

## adaBoost.R

Implements AdaBoost.M1.

#### Public function:
	adaBoost()

## adaBoost_spam.R

Compares the AdaBoost.M1 implementation against 'gbm'.

#### Public function:
	adaBoost_spam.R()

## Packages required:
	gbm
	rpart
	ggplot2
	assertthat
	caTools

When run directly this R script will execute the adaBoost_spam() function which will create a PDF
and a PNG containing its test results.  A relatively low depth of 8 is permitted for the classification
trees in order to force the predictors to struggle.  With adequate depth, the performance of both predictors is
excellent and therefore less good for comparison.
