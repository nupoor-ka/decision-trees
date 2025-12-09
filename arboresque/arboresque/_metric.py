import pandas as pd
import numpy as np

"""
gini, entropy, log loss
mae, mse, friedman mse, poisson
the process involves testing different splits for impurity reduction
the split which provides maximum reduction in impurity is chosen
homogeneity is high if measure is low
"""

def gini(y) -> float: # can bring in another function to keep state later, for optimization
    """gini impurity, classification"""
    y = np.asarray(y)
    counts = np.bincount(y) # assuming y vals are non-negative ints
    class_counts_ = counts.astype(float) # got class counts
    n_samples_ = y.shape[0]
    if n_samples_ == 0: # preventing div by 0 err
        return 0.0
    p = class_counts_ / n_samples_ # got class proportions
    return 1.0 - np.sum(p ** 2) # gini formula

def entropy(y):
    """
    entropy, classification
    equivalent to measuring log loss between ground truth and prediction
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    counts = np.bincount(y)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0] # mask zeros to avoid log(0)
    return -np.sum(p * np.log2(p))

def mse(y):
    """
    mean squared error, regression
    equivalent to working with variance
    same as squared error option in sk learn
    assuming a gaussian distribution intuitively
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)

friedman = mse # for single trees, mostly similar

def mae(y):
    """mean absolute error, regression"""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    median = np.median(y) # different from others
    return np.mean(np.abs(y - median))

def poisson_deviance(y):
    """
    difference in log likelihood of correct prediction based on target model vs based on predicted model
    y is some sort of count or frequency, non negative
    the higher the mean, the greater the variance
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    mean = np.mean(y) # model prediction for all samples in this node
    if mean <= 0:
        return 0.0
    mask = y > 0 # can't compute for negative vals
    deviance = 2 * np.sum(y[mask] * np.log(y[mask] / mean) - (y[mask] - mean)) # sk learn doesn't double
    n_zeros = np.sum(~mask) # contri from terms equal to 0
    deviance += 2 * n_zeros * mean
    return deviance / y.size # normalize by sample count, mean of diff in ll
