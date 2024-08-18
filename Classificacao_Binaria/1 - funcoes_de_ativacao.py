import numpy as np
# site: keras.io, para aprender mais sobre outras funções de ativação


def step_function(soma):  # transfer function
    if soma >= 1:
        return 1
    return 0


def sigmoid_function(soma):
    return 1 / (1 + np.exp(-soma))


def tanh_function(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))


def relu_function(soma):
    if soma >= 0:
        return soma
    return 0


def linear_function(soma):
    return soma


def softmax_function(x):  # x = [7.0, 2.0, 1.3], quanto maior o valor, maior a probabilidade.
    ex = np.exp(x)
    return ex / ex.sum()
