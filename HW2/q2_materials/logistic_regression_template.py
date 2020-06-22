import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
from plot_digits import *
import matplotlib.pyplot as plt

def run_logistic_regression(l, w, i, p, penalty=False):
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': l,
                    'weight_regularization': w,
                    'num_iterations': i,
                    'penalty': p
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.normal(0, hyperparameters['weight_regularization'], M+1).reshape(M+1, 1) # unsure vectors are in (R,1) shape
    
    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    #run_check_grad(hyperparameters)
    ce_train_list = []
    ce_val_list = []
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
        ce_train_list.append(cross_entropy_train[0][0])
        
        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")
        
        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N
        
        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        pred_test = logistic_predict(weights, test_inputs)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, pred_test)
        
        ce_val_list.append(cross_entropy_valid[0][0])
        
        #print some stats
        if t + 1 == hyperparameters['num_iterations']:
            print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train*100),
                   float(cross_entropy_valid), float(frac_correct_valid*100))
            print "TEST CE=", cross_entropy_test, "acc=", frac_correct_test
            #return cross_entropy_train, 1 - frac_correct_train, cross_entropy_valid, 1 -frac_correct_valid
    
    return ce_train_list, ce_val_list

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
    #for l in [1.0, 0.1, 0.01, 0.001]:
        #for w in [1.0, 0.1, 0.01, 0.001]:
            #for i in [50, 100, 150, 200, 250]:
                #for p in [0.001, 0.01, 0.1, 1.0]:
                    #print "Running logistic regression with learning rate=", l,\
                              #", weight regularization=", w, ", number of iterations=", i, 'penalty=', p
                    #run_logistic_regression(l, w, i, p, penalty=True)
    train_ce, val_ce = run_logistic_regression(1.0, 1.0, 100, 1.0, penalty=False)
    
    #sum_t_ce = 0
    #sum_v_ce = 0
    #sum_t_fe = 0
    #sum_v_fe =0
    #sum_t_ce, sum_t_fe, sum_v_ce, sum_v_fe = run_logistic_regression(1.0, 1.0, 100, 0, penalty=False)
    #list0 = []
    #list1 = []
    #list2 = []
    #list3 = []
    #ps = [0.001, 0.01, 0.1, 1.0]
    #for p in ps:
        #for i in range(5):
            #if i == 0:
                #sum_t_ce, sum_t_fe, sum_v_ce, sum_v_fe = run_logistic_regression(
                    #1.0, 1.0, 100, p, penalty=False)
            #else:
                #temp0, temp1, temp2, temp3 = run_logistic_regression(1.0, 1.0, 100, p, penalty=True)
                #sum_t_ce = sum_t_ce + temp0
                #sum_t_fe = sum_t_fe + temp1 * 100
                #sum_v_ce = sum_v_ce + temp2
                #sum_v_fe = sum_v_fe + temp3 * 100      
        #list0.append((sum_t_ce / 5)[0][0])
        #list1.append(sum_t_fe / 5)
        #list2.append((sum_v_ce / 5)[0][0])
        #list3.append(sum_v_fe / 5)
    
    #ps = ['0.001', '0.01', '0.1', '1.0']
    #plt.plot(ps,list0, label='Train entropy')
    #plt.plot(ps,list2, label='Validation entropy')
    #plt.title('Cross Entropy over penalty lambda')
    #plt.xlabel('Penalty')
    #plt.ylabel('Cross Entropy')
    #plt.legend()
    #plt.savefig("ce" + str(0.1) + ".png")
    #plt.clf()    
    #plt.plot(ps,list1, label='Train error')
    #plt.plot(ps,list2, label='Validation error')
    #plt.title('Classification error over penalty')
    #plt.xlabel('Penalty')
    #plt.ylabel('Classification')
    #plt.legend()
    #plt.savefig("ce" + str(0.2) + ".png")
    #plt.clf()      
    
    plt.plot(train_ce, label='Train CE')
    plt.plot(val_ce, label='Validation CE')
    plt.title('Cross Entropy changes')
    plt.xlabel('Iteration')
    plt.ylabel('Classification')
    plt.legend()
    plt.savefig("ce" + str(1) + ".png")
    plt.clf()        
    
