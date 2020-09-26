import inspect

import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def relu(z):
    return np.maximum(z, 0)


def gaussian_black(z, mu: 'normal' = 0, sigma: (0.4,1) = 0.7):
    """
    Possibly useful on output layer (David Ha uses)
    Can increase sigma to get lighter shades.
    Sigma of 0.5 gives more black background + neon effect
    """
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power((z - mu)/sigma, 2)/2)


def gaussian_white(z, mu: 'normal' = 0, sigma: (0.4, 1) = 0.7):
    """
    One minus the gaussian, meaning most values come out white rather than black.
    """
    return 1 - gaussian_black(z, mu, sigma)


def modz(z, divisor=1, thresh=0.25):
    remainders = np.mod(z, divisor)
    return (remainders<thresh).astype(float)


def round1dp(z, decimals: (0,3) = 1):
    return np.round(z, decimals=decimals)


def point(z, p=0, thresh=0.05):
    return (np.abs(z-p)<thresh).astype(float)


# Wrapping some numpy functions so that my create_args works
def tanhz(z):
    return np.tanh(z)


def sinz(z):
    return np.sin(z)


def absz(z):
    return np.abs(z)


def nofunc(z):
    return z


def get_funcs(func_name):
    func_dict = {
        'nofunc': nofunc,
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': tanhz,
        'sin' : sinz,
        'abs': absz,
        'gaussian_black': gaussian_black,
        'gaussian_white': gaussian_white,
        'mod': modz,
        'round': round1dp,
        'point': point
        }
    if func_name=='names':
        # Special useage to get all the opotions.
        return list(func_dict.keys())
    else:
        return func_dict[func_name]

    
def create_args(func):
    """
    For a given function it returns a dict of random args by 
    peturbing the defaults.
    Does not return the arg called 'z' which we will always
    use for the vector input.
    Assumes all other args are scalar floats.
    """
    # Get a dictionary of the params of the function
    params = dict(inspect.signature(func).parameters)
    # We will always use z for the vector input so delete that from the dict
    del params['z']
    return {k: peturb(v) for k, v in params.items()}
    
def peturb(param):
    """
    Returns a peturbed version of the default parameter value.
    But becuase certain args must meet certain requirements
    (eg. sigma in gaussian() must be positive), the args
    in the original functions can be annotated to indicate
    these conditions. This function interprets those annotations
    and peturbs accordingly.
    """
    ann = param.annotation
    if ann == inspect._empty:
        ann = 'normal'
    if type(ann)==str:
        if ann == 'normal':
            return param.default + np.random.normal()
        elif ann == 'positive':
            return abs(param.default + np.random.normal())
    elif type(ann) == tuple:
        # Get a number from uniform random distribution
        # bounded by values in the annotation tuple.
        if type(ann[0]) == float:
            return np.random.uniform(*ann)
        elif type(ann[0]) == int:
            return np.random.randint(*ann)
    else:
        print('Unrecognised function annotation.')
