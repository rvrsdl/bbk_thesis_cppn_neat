import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def relu(z):
    return np.maximum(z,0)

def get_funcs(func_name):
    func_dict = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': np.tanh
        }
    return func_dict[func_name]