import numpy as np
from numpy.linalg import pinv

def shuffle_data(data):
    """ Takes in a data that contains a (t,X) pair and returns its randomly
    permuted version along the samples. """
    # generate an array that contains a random order of indices for data
    random_order = np.random.permutation(len(data["X"]))
    # return de reordered X and t
    return {"X": data["X"][random_order], "t": data["t"][random_order]}

def split_data(data, num_fold, fold):
    """
    Takes in data that is a (t,X) pair, folds each row into num_fold equal folds
    and the foldth will be the validation fold. Returns the validation fold as 
    a 1d array and the rest folds as a 2d array.
    """
    # fold the samples and target
    folded_x = np.array_split(data["X"], num_fold)
    folded_t = np.array_split(data["t"], num_fold)
    # pop the fold to be used as validation set
    data_fold = {"X": folded_x.pop(fold-1), "t": folded_t.pop(fold-1)}
    # add the rest of the folds into return value
    data_rest = {"X":np.concatenate(folded_x), "t":np.concatenate(folded_t)}
    return data_fold, data_rest


def train_model(data, lambd):
    """
    Takes a set of data and return the coefficient of rideg regression with
    penalty level lambd.
    """
    x = np.array(data["X"])
    t = data["t"]
    x_trans = x.transpose()
    return np.matmul(pinv(np.matmul(x_trans, x) +
             np.dot(lambd,np.identity(len(x_trans)))), np.matmul(x_trans, t))

def predict(data, model):
    """ Takes in data and model coefficient and predict the outcome. """
    return np.matmul(data["X"], model)

def loss(data, model):
    """ Takes in the data and model, and return the average sqaured error loss.
    """
    t_xw = data["t"] - predict(data, model)
    return t_xw.dot(t_xw) / len(data["t"])


def cross_validation(data, num_folds, lambd_seq):
    """ Takes in the training data, number of folds, and a sequence of ¦Ë and
    return a vector of 50 cross validation erros. """
    cv_error = []
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds+1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error

if __name__ == "__main__":
    data_train = {"X": np.genfromtxt("data_train_X.csv", delimiter=','), 
                  "t": np.genfromtxt("data_train_y.csv", delimiter=',')}
    data_test = {"X": np.genfromtxt("data_test_X.csv", delimiter=','),
                 "t": np.genfromtxt("data_test_y.csv", delimiter=',')}
    lambd_seq = np.linspace(0.02, 1.5, num=50)
    
    #errors = cross_validation(data_train,10, lambd_seq)
    #print(errors)