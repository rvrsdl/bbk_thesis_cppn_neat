"""
Created on Thu Jul 23 18:20:00 2020

@author: riversdale

Functions for converting pixel coordinates into
Fourier features.
Mainly copied from: 
    https://github.com/noahtren/Fourier-Feature-Networks-TensorFlow-2
"""
import numpy as np

# def get_coord_ints(y_dim, x_dim):
#   """Return a 2x2 matrix where the values at each location are equal to the
#   indices of that location
#   """
#   ys = tf.range(y_dim)[tf.newaxis]
#   xs = tf.range(x_dim)[:, tf.newaxis]
#   coord_ints = tf.stack([ys+xs-ys, xs+ys-xs], axis=2)
#   return coord_ints


# def generate_scaled_coordinate_hints(batch_size, img_dim=256):
#   """Generally used as the input to a CPPN, but can also augment each layer
#   of a ConvNet with location hints
#   """
#   spatial_scale = 1. / img_dim
#   coord_ints = get_coord_ints(img_dim, img_dim)
#   coords = tf.cast(coord_ints, tf.float32)
#   coords = tf.stack([coords[:, :, 0] * spatial_scale,
#                      coords[:, :, 1] * spatial_scale], axis=-1)
#   coords = tf.tile(coords[tf.newaxis], [batch_size, 1, 1, 1])
#   return coords

# RW
def get_coords(img_size=(64,64)):
    x = np.linspace(-1, 1, img_size[0])
    y = np.linspace(-1, 1, img_size[1])
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx,yy], axis=-1)
    return coords
    

def initialize_fourier_mapping_vector(n_features=256, sigma=10):
  dims = 2
  #B = tf.random.normal((m, d)) * sigma
  B = np.random.normal(size=(n_features,dims), scale=sigma)
  return B


def fourier_mapping(coords, B):
  """
  Preprocess each coordinate — scaled from [0, 1) — by converting each
  coordinate to a random fourier feature, as determined by a matrix with values
  samples from a Gaussian distribution.
  """
  #sin_features = tf.math.sin((2 * math.pi) * (tf.matmul(coords, B, transpose_b=True)))
  sin_features = np.sin((2*np.pi) * (coords @ B.T))
  #cos_features = tf.math.cos((2 * math.pi) * (tf.matmul(coords, B, transpose_b=True)))
  cos_features = np.cos((2*np.pi) * (coords @ B.T))
  #features = tf.concat([sin_features, cos_features], axis=-1)
  features = np.concatenate([sin_features, cos_features], axis=-1)
  # Should have shape: img_size[0], img_size[1], n_features*2
  return features