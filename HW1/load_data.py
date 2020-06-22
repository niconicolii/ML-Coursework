from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import math
from collections import Counter

def load_data():
    """ Load headlines from clean_real.txt and clean_fake.txt and seperate them
    into training set, validation set, and testing set. """
    # read clean_fake.txt and store each headline as an element into an array
    fake_file = open("clean_fake.txt", "r")
    fake_arr = fake_file.read().split('\n')
    fake_file.close()
    # read clean_real.txt and store each headline as an element into an array
    real_file = open("clean_real.txt", "r")
    real_arr = real_file.read().split('\n')
    real_file.close()
    #combine the two arrays into one sample array
    sample_arr = fake_arr + real_arr
    # target array, 0 means corresponding headline is fake, 1 means real
    target_arr = [0] * len(fake_arr) + [1] * len(real_arr)
    # split into three sets
    vectorizer = CountVectorizer()
    sample = vectorizer.fit_transform(sample_arr)
    x_train, x_test, y_train, y_test = train_test_split(
        sample.toarray(), target_arr, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5)
    #return the dictionary
    return {"train":x_train, "train_target": y_train, "val":x_val,
            "val_target":y_val,  "test":x_test, "test_target": y_test,
            "feature_names":vectorizer.get_feature_names()}

def select_model():
    """
    Select the model with the most efficient hyperparameter.
    """
    # get data
    data = load_data()
    # initialize the variables for storing the model with best accuracy
    best_acc = 0
    best_depth = 0
    best_crit = ""
    # two criterion to choose
    for crit in ["gini", "entropy"]:
        # 5 different types of maximum depth of the tree
        for i in [4,8,12,16,20]:
            # train a tree with given criterion and depth
            clf = DecisionTreeClassifier(criterion=crit, max_depth=i)
            clf.fit(data["train"], data["train_target"])
            # accuracies testing on the validation set, store best accuracy 
            acc = clf.score(data["val"], data["val_target"])
            (best_acc, best_depth, best_crit) = (
                best_acc, best_depth, best_crit
                ) if best_acc > acc else (acc, i, crit)
            print("Score (max_depth=", i, ", criterion=", crit, ") :", acc)
    print("Best Model(max_depth)=" , best_depth, ", criterion=", 
          best_crit, "): Score=", best_acc)
def n_nlogn(n):
    """ calculate negative n times log base 2 of n """
    return (-1 * n * math.log2(n)) if n > 0 else 0


def compute_information_gain(keyword):
    data = load_data() 
    total = len(data["train_target"])   # total number of news
    # get the index of keyword in the list of features
    index = data["feature_names"].index(keyword)
    train = data["train"]
    real = 0            # number of real news
    real_key = 0        # number of real news with keyword
    key = 0             # number of news with keyword
    for i in range(total):
        if train[i][index] > 0:
            key += 1                        # count one news with keyword
            if data["train_target"][i] == 1 : 
                real_key += 1               # count one real news with keyword
        if data["train_target"][i] == 1:
            real += 1                       # count one real news
    entropy = n_nlogn(real / total) + n_nlogn(
        (total - real)/ total)          # H(Y)
    p_e_r = real_key / key              # real news, with keyword
    p_e_f = (key-real_key) / key        # fake news, with keyword
    p_ne_r = (real-real_key) / (total-key)              # real news, no keyword
    p_ne_f = (total-real-(key-real_key)) / (total-key)  # fake news, no keyword
    entropy_keyword = (key/total) * (n_nlogn(p_e_r) + n_nlogn(p_e_f)) + (
        (total-key)/total) * (n_nlogn(p_ne_r) + n_nlogn(p_ne_f))    # H(Y|xi)
    IG = entropy - entropy_keyword      # information gain
    print("Information Gain in spliting at keyword '", keyword, "' is ", IG)

if __name__ == "__main__":
    for key in ["donald", "trumps", "hillary", "the", "trump",
                "breaking", "prime"]:
        compute_information_gain(key)
    