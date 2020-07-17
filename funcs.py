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

def get_mod_func():
    divisor = np.random.randn(1) # NAAAT ok to have randomness here. Same genome should always produce same net.
    thresh = np.random.randn(1) # more hit and miss having thresh random rather than median.
    divisor = 2
    thresh = 1 #with div=2, about 50% of samples from random normal will be below 1.
    # so thresh kinda corresponds to the thickness of the line/stripe. <0.1 gives thin line but <1 gives equal stripes
    # so could keep div constant at 2 and select thresh from uniform random dist between 0.1 and 1
    def mod_func(z):
        remainders = np.mod(z, divisor)
        #thresh = np.median(remainders)
        return (remainders<thresh).astype(float)
    return mod_func

def round1dp(z):
    return np.round(z, decimals=1)

def get_funcs(func_name):
    func_dict = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': np.tanh,
        'sin' : np.sin,
        'abs': np.abs,
        'gaussian': gaussian,
        'mod': get_mod_func(),
        'round': round1dp
        }
    if func_name=='names':
        # Special useage to get all the opotions.
        return list(func_dict.keys())
    else:
        return func_dict[func_name]