import numpy as np
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt
import os

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

m = train.shape[1]

def layer_sizes(train,test):
    n_x = train.shape[0]
    n_y = test.shape[0]
    n_h = 4
    return (n_x,n_y,n_h)


def initialize_parameters(n_x, n_y, n_h):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros([n_h, 1])
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros([n_y, 1])

    parameters = {"w1": w1,
                  "w2": w2,
                  "b1": b1,
                  "b2": b2}
    return parameters


def foreward_propagation(x, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    z1 = np.dot(w1, train) + b1
    A1 = np.tanh(z1)
    z2 = np.dot(w2, A1) + b2
    A1 = np.sigmoid(z2)

    cache = {"z1": z1,
             "z2": z2,
             "A1": A1,
             "A2": A2}

    return A2, cache


def compute_cost(A2, test, parameters):
    m = test.shape[1]
    logprobs = np.multiply(np.log(A2), test) + np.multiply((1 - test), np.log(1 - A2))
    cost = (-1 / m) * np.sum(logprobs)

    return cost


def backward_propagation(cache, parameters, train, test):
    m = train.shape[1]

    w1 = parameters["w1"]
    w2 = parameters["w2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - test
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # sum through the columns No_of rows stays same
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, train.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    n_x = layer_sizes(train, test)[0]
    n_y = layer_sizes(train, test)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, train):
    A2, cache = forward_propagation(train, parameters)
    predictions = np.round(A2)

    return predictions


predictions = predict(parameters, train)
print('Accuracy: %d' % float(
    (np.dot(test, predictions.T) + np.dot(1 - test, 1 - predictions.T)) / float(test.size) * 100) + '%')
