""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    y = sigmoid(np.dot(data, weights[:-1]) + weights[-1]) 
    return y.reshape((len(y), 1))   # reshape to keep vector's shape consistent

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    ce = np.dot(targets.T, -1 * np.log(y))
    frac_correct = np.mean((y > 0.5) == targets)
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """
    N = len(targets)
    # TODO: Finish this function
    y = logistic_predict(weights, data)
    z = np.dot(data, weights[:-1]) + weights[-1]
    f = np.dot(targets.T, -1 * np.logaddexp(0, -1*z)) + np.dot((1-targets).T, np.logaddexp(0,z))
    # add one more column of ones for bias
    data_ca = np.concatenate((data, np.ones((N, 1))), axis=1)
    df = np.dot(data_ca.T, y-targets)
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # TODO: Finish this function
    N = len(targets)
    reg = hyperparameters['penalty']
    lf, ldf, y = logistic(weights, data, targets, hyperparameters)
    
    f = lf + ((reg/2) * np.dot(weights.T, weights)) - (np.log(reg/(2*np.pi))*N/2)
    df = ldf + (reg * weights)
    return f, ldf, y
