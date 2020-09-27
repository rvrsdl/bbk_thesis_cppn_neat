"""
Created on Thu Jul 23 18:20:00 2020

@author: riversdale

Functions for converting pixel coordinates into
Fourier features.
Mostly reused code from:
    https://github.com/noahtren/Fourier-Feature-Networks-TensorFlow-2
"""
import numpy as np

def get_coords(img_size=(64,64)):
    x = np.linspace(-1, 1, img_size[0])
    y = np.linspace(-1, 1, img_size[1])
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx,yy], axis=-1)
    return coords
    

def initialize_fourier_mapping_vector(n_features: int = 256, 
                                      sigma: float = 10.0) -> np.ndarray:
  dims = 2
  B = np.random.normal(size=(n_features,dims), scale=sigma)
  return B


def fourier_mapping(coords: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Preprocess each coordinate — scaled from [0, 1) — by converting each
  coordinate to a random fourier feature, as determined by a matrix with values
  samples from a Gaussian distribution.
  """
  sin_features = np.sin((2*np.pi) * (coords @ B.T))
  cos_features = np.cos((2*np.pi) * (coords @ B.T))
  features = np.concatenate([sin_features, cos_features], axis=-1)
  # Should have shape: img_size[0], img_size[1], n_features*2
  return features