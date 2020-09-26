# Testing the things in the funcs.py file.
# Most of the underlying functions point directly to
# numpy funcs so no need to test those ones.
import inspect

import numpy as np

from src.funcs import get_funcs, create_args

def test_create_args():
    """
    Test the creat_args function which creates
    random parameters for functions which require them
    """
    all_func_names = get_funcs('names')
    for fname in all_func_names:
        f = get_funcs(fname)
        arg_dict = create_args(f)
        expected_args = dict(inspect.signature(f).parameters)
        del expected_args['z'] # We don't create an argument for the input vector
        for a in expected_args:
            assert a in arg_dict

def test_mod():
    """
    Test the new thresholded mod function.
    """
    f = get_funcs('mod')
    arg_dict = create_args(f)
    in_vec = np.arange(-10, 10, 0.2)
    out_vec = f(in_vec, **arg_dict)
    expected = (np.mod(in_vec, arg_dict['divisor']) < arg_dict['thresh']).astype(float)
    assert np.all(out_vec == expected)

def test_round():
    """
    Test the round function
    """
    f = get_funcs('round')
    arg_dict = create_args(f)
    in_vec = np.arange(-10, 10, 0.2)
    out_vec = f(in_vec, **arg_dict)
    expected = np.round(in_vec, arg_dict['decimals'])
    assert np.all(out_vec == expected)