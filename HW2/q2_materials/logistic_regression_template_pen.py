import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *

def run_logistic_regression(l, w, i, penalty=False):
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': l,
                    'weight_regularization': w,
                    'num_iterations': i
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.normal(0, 0.1, M+1).reshape(M+1, 1)# unsure vectors are in (R,1) shape
    
    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    #run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    for t in xrange(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        if penalty:
            f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        else:
            f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
        
        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")
        
        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N
        
        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        # print some stats
        if t + 1 == hyperparameters['num_iterations']:
            print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train*100),
                   float(cross_entropy_valid), float(frac_correct_valid*100))

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)
    
    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    for l in [1.0, 0.1, 0.01, 0.001]:
        for w in [1.0, 0.1, 0.01, 0.001]:
            for i in [100, 400, 800, 1000]:
                print "Running logistic regression with learning rate=", l,\
                          ", weight regularization=", w, ", number of iterations=", i
                run_logistic_regression(l, w, i, penalty=True)