from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()
import numpy as np
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def train_mle_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MLE estimators theta_mle and pi_mle"""

    # YOU NEED TO WRITE THIS PART
    (image_num, pixel_num) = np.shape(train_images)
    class_num = np.shape(train_labels)[1]
    
    theta_mle = np.zeros((pixel_num, class_num))
    pi_mle = np.zeros(class_num)
    
    for c in range(class_num):
        tc = train_labels[:, c].copy()
        for j in range(pixel_num):
            xj = train_images[:, j].copy()
            theta_mle[j,c] = np.sum(tc * xj) / np.sum(tc)
        if(c == 9):
            pi_mle[c] = 2 * np.sum(tc) / (image_num + np.sum(train_labels[:, 9]))
        else:
            pi_mle[c] = np.sum(tc) / (image_num + np.sum(train_labels[:, 9]))
    
    return theta_mle, pi_mle


def train_map_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MAP estimators theta_map and pi_map"""
    
    # YOU NEED TO WRITE THIS PART
    (image_num, pixel_num) = np.shape(train_images)
    class_num = np.shape(train_labels)[1]
    
    theta_map = np.zeros((pixel_num, class_num))
    pi_map = np.ones(class_num)
    
    for c in range(class_num):
        tc = train_labels[:, c].copy()
        for j in range(pixel_num):
            xj = train_images[:, j].copy()
            theta_jc = (2 + np.sum(tc * xj)) / (4 + np.sum(tc))
            theta_map[j][c] = theta_jc
        if(c == 9):
            pi_map[c] = 2 * np.sum(tc) / (image_num + np.sum(train_labels[:, 9]))
        else:
            pi_map[c] = np.sum(tc) / (image_num + np.sum(train_labels[:, 9]))
   
    return theta_map, pi_map


def log_likelihood(images, theta, pi):
    """ Inputs: images, theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
    log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
    log_like is a matrix of num of images x num of classes
    Note that log likelihood is not only for c^(i), it is for all possible c's."""

    # YOU NEED TO WRITE THIS PART
    image_num = np.shape(images)[0]
    class_num = np.shape(theta)[1]
    
    log_like = np.zeros((image_num, class_num))
    for i in range(image_num):
        for c in range(class_num):
            nume = pi[c] * np.prod(np.power(theta[:,c], images[i]) * np.power(1-theta[:,c], 1-images[i]))
            if(nume != 0):
                denom = 0
                for cc in range(class_num):                    
                    denom += pi[cc] * np.prod(np.power(theta[:,cc], images[i]) * np.power(1-theta[:,cc], 1-images[i]))
                log_like[i][c] = np.log(nume / denom)
            else:
                log_like[i][c] = 0
    
    return log_like


def predict(log_like):
    """ Inputs: matrix of log likelihoods
    Returns the predictions based on log likelihood values"""

    # YOU NEED TO WRITE THIS PART
    image_num = np.shape(log_like)[0]
    predictions = np.zeros(np.shape(log_like))
    for i in range(image_num):
        max_index = np.argmax(log_like[i])
        predictions[i][max_index] = 1
    
    return predictions


def accuracy(log_like, labels):
    """ Inputs: matrix of log likelihoods and 1-of-K labels
    Returns the accuracy based on predictions from log likelihood values"""

    # YOU NEED TO WRITE THIS PART
    predictions = predict(log_like)
    match_num = 0
    N = np.shape(labels)[0]
    for i in range(N):
        if(np.array_equal(predictions[i], labels[i])):
            match_num += 1
    accuracy = match_num / N
        
    return accuracy


def image_sampler(theta, pi, num_images):
    """ Inputs: parameters theta and pi, and number of images to sample
    Returns the sampled images"""
    
    # YOU NEED TO WRITE THIS PART
    (pixel_num, class_num) = np.shape(theta)
    
    sampled_images = np.zeros((num_images, pixel_num))
    cs = np.random.choice(class_num, num_images, p=pi)
    for i in range(num_images):
        sampled_images[i] = np.random.binomial(1, p=theta[:, cs[i]])
    print("The random classes are: " + str(cs))
    return sampled_images


def main():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    # Fit MLE and MAP estimators
    theta_mle, pi_mle = train_mle_estimator(train_images, train_labels)
    theta_map, pi_map = train_map_estimator(train_images, train_labels)

    # Find the log likelihood of each data point
    loglike_train_mle = log_likelihood(train_images, theta_mle, pi_mle)
    loglike_train_map = log_likelihood(train_images, theta_map, pi_map)

    avg_loglike_mle = np.sum(loglike_train_mle * train_labels) / N_data
    avg_loglike_map = np.sum(loglike_train_map * train_labels) / N_data

    print("Average log-likelihood for MLE is ", avg_loglike_mle)
    print("Average log-likelihood for MAP is ", avg_loglike_map)

    train_accuracy_map = accuracy(loglike_train_map, train_labels)
    loglike_test_map = log_likelihood(test_images, theta_map, pi_map)
    test_accuracy_map = accuracy(loglike_test_map, test_labels)

    print("Training accuracy for MAP is ", train_accuracy_map)
    print("Test accuracy for MAP is ", test_accuracy_map)

    # Plot MLE and MAP estimators
    save_images(theta_mle.T, 'mle.png')
    save_images(theta_map.T, 'map.png')

    # Sample 10 images
    sampled_images = image_sampler(theta_map, pi_map, 10)
    save_images(sampled_images, 'sampled_images.png')


if __name__ == '__main__':
    main()
