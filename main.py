#!/usr/bin/python3.4
# -*-coding:Utf-8 -*

import scipy.io
import scipy.optimize
import numpy as np


def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12

    randomArray = np.random.rand(L_out, 1+L_in)

    W = randomArray * 2 * epsilon_init - epsilon_init

    return W


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidGrad(x):
    return sigmoid(x) * (1 - sigmoid(x))


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels,
                   X, y, lambdaa):
    """
    Fonction cout, retourne resultat et gradient
    """
    Theta1 = np.reshape(nn_params[:(hidden_layer_size *
                                    (input_layer_size + 1))],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size *
                                  (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    m = X.shape[0]

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    delta1 = np.zeros(Theta1.shape)
    delta2 = np.zeros(Theta2.shape)

    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    for i in range(m):
        Xi = X[i, :].T
        yi = y[i]
        y_vect = np.zeros((num_labels, 1))

        if (yi == 0):
            y_vect[9] = 1
        else:
            y_vect[yi - 1] = 1

        z2 = np.dot(Theta1, Xi)
        a2 = sigmoid(z2)
        a2 = np.concatenate((np.ones(1), a2), axis=0)
        z3 = np.dot(Theta2, a2)
        a3 = sigmoid(z3)

        J += np.sum(-y_vect * np.log(a3) - (1 - y_vect)*np.log(1 - a3))

        little_delta_3 = a3 - y_vect

        print(np.dot(Theta2.T, little_delta_3))

        print(sigmoidGrad(np.concatenate((np.ones(1), z2), axis=0)).shape)

        little_delta_2 = np.dot(Theta2.T, little_delta_3) * \
            sigmoidGrad(np.concatenate((np.ones(1), z2), axis=0))
        little_delta_2 = little_delta_2[1:]
        delta1 = delta1 + np.dot(little_delta_2, Xi.T)
        delta2 = delta2 + np.dot(little_delta_3, a2.T)

    J = J/m

    temp1 = Theta1
    temp1[:, 0] = np.zeros(temp1.shape[0], 1)
    temp2 = Theta2
    temp2[:, 0] = np.zeros(temp2.shape[0], 1)

    J += lambdaa / (2*m) * (np.sum(temp1*temp1) + np.sum(temp2*temp2))

    Theta1_grad = delta1/m + temp1*lambdaa/m
    Theta2_grad = delta2/m + temp2*lambdaa/m

    grad = np.concatenate(((Theta1_grad.T).ravel(), (Theta2_grad.T).ravel()))

    return (J, grad)


# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10

mat = scipy.io.loadmat('ex4data1.mat')
X = mat['X']
y = mat['y']

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.concatenate(((initial_Theta1.T).ravel(),
                                    (initial_Theta2.T).ravel()))


lambdaa = 1


def costFunction(p):
    J, grad = nnCostFunction(p, input_layer_size,
                             hidden_layer_size, num_labels,
                             X, y, lambdaa)
    return J


def gradCostFunction(p):
    J, grad = nnCostFunction(p, input_layer_size,
                             hidden_layer_size, num_labels,
                             X, y, lambdaa)
    return grad


nn_params = scipy.optimize.fmin_cg(costFunction, initial_nn_params,
                                   gradCostFunction, maxiter=5)
