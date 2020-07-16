import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def relu(z):
    return np.maximum(z,0)

def gaussian(z, mu=0, sigma=1):
    """
    Possibly useful on output layer (David Ha uses)
    Can increase sigma to get lighter shades.
    """
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power((z - mu)/sigma, 2)/2)

def get_funcs(func_name):
    func_dict = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': np.tanh,
        'sin' : np.sin,
        'abs': np.abs,
        'gaussian': gaussian
        }
    if func_name=='names':
        # Special useage to get all the opotions.
        return list(func_dict.keys())
    else:
        return func_dict[func_name]