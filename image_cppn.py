"""
Created on Wed Jul 29 22:33:02 2020

@author: riversdale

Has stuff to do with processing i
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from matplotlib.image import imsave, imread
import matplotlib.pyplot as plt

from nnet import NNFF
import fourier

# typing definitions
Dims = Tuple[int, int]

class Image:
    """
    Simple class to hold image data and some metadata
    like size and number of channels.
    Might also have save and show methods etc.
    """
    
    def __init__(self, data : np.ndarray):
        shape = data.shape
        if len(shape) == 2:
            self.channels = 1
            self.size = shape
        elif len(shape) == 3:
            self.channels = shape[2]
            self.size = shape[:-1]
        else:
            raise Exception("Invalid array shape")
        assert np.max(data) <= 1 and np.min(data) >=0, "Image data must be in range 0-1"
        self.data = data
            
    def save(self, filename: str) -> None:
        """
        Saves the image.
        """
        # Note the cmap param is ignored if we have RGB data.
        imsave(filename, self.data, vmin=0, vmax=1, cmap='gray')
        
    @staticmethod
    def load(filename: str, channels=3) -> Image:
        """
        Reads data from a saved PNG
        Useful for loading target image
        """
        img = imread(filename)
        img = np.squeeze(img[:,:,:channels])
        return Image(img)
        # Do we need to rescale values to be 0-1 ??
    
    def show(self) -> None:
        """
        Shows the image in the plots window.
        """
        if self.channels==1:
            plt.imshow(self.data, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(self.data, vmin=0, vmax=1)
        plt.show()
        
    def diff(self, shift=1) -> Image:
        """
        Subtracts pixel values from vals shifted one left and one down
        """
        # np.roll goes round the back which isn't really relevant so just put zero diff for those pixels
        abs_diff = np.abs(self.data - np.roll(self.data, shift=shift, axis=(0,1)))
        if shift>0:
            abs_diff[:shift,:] = 0
            abs_diff[:,:shift] = 0
        else:
            # In case of negative shift
            abs_diff[shift:,:] = 0
            abs_diff[:,shift:] = 0
        return Image(abs_diff)
            

class CPPN:
    
    def __init__(self, net: NNFF, fourier_vec : np.ndarray = None):
        self.net = net
        self.channels = net.n_out
        self.fourier_vec = fourier_vec
        if fourier_vec is None:
            self.bias_vec_len = net.n_in - 3
        else:
            self.bias_vec_len = net.n_in - len(fourier_vec)*2           
    
    def get_coords(self, imsize: Dims) -> np.ndarray: 
        x = np.linspace(-1, 1, imsize[0])
        y = np.linspace(-1, 1, imsize[1])
        xx, yy = np.meshgrid(x, y)
        coords = np.stack([xx,yy], axis=-1)
        return coords
    
    def create_image(self, imsize: Dims = (128,128), bias=None) -> Image:
        """
        Single-pass image creation which should be able to cope with
        vanilla or fourier approach, and can have a bias vector of any length.
        """
        if bias is None:
            bias = np.tile(1, self.bias_vec_len)
        assert len(bias)==self.bias_vec_len, "Bias vector was an unexpected size."
        coords = self.get_coords(imsize)
        pixels = imsize[0]*imsize[1]
        bias_tile = np.tile(bias, (pixels,1)) # The bias (scalar or vector must go in for every pixel)
        if self.fourier_vec is None:
            # Vanilla CPPN with coordinate inutes to neural network
            xcoords= coords[:,:,0].ravel()
            ycoords= coords[:,:,1].ravel()
            dcoords = np.sqrt(xcoords**2 + ycoords**2)
            img_raw = np.array(self.net.feedforward((xcoords, ycoords, dcoords, *bias_tile.T)))
        else:
            # Use fourier features rather than coordinates as inputs to the neural net.
            feats = fourier.fourier_mapping(coords, self.fourier_vec)
            n_ffeats = self.fourier_vec.shape[0]
            feats = feats.reshape(pixels, n_ffeats*2) # times two because we have sin and cos features
            img_raw = np.array(self.net.feedforward((*feats.T, *bias_tile.T)))
        if self.channels==1:
            img_square = img_raw.T.reshape(imsize[0], imsize[1])
        else:
            img_square = img_raw.T.reshape(imsize[0], imsize[1], self.channels)
        return Image(img_square)
