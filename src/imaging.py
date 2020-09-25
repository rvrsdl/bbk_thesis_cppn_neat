"""
Created on Wed Jul 29 22:33:02 2020

@author: riversdale

Has stuff to do with processing i
"""
from __future__ import annotations
from typing import Tuple, List, Dict
import os
from datetime import datetime

import numpy as np
from matplotlib.image import imsave, imread
import matplotlib.pyplot as plt

from src.nnet import NNFF
from src import fourier
from src.genome import Genome
from src.perlin import get_perlin_noise

# typing definitions
Dims = Tuple[int, int]

class ImageCreator(object):
    
    def __init__(self, save_loc: str = os.getcwd(), colour_channels: int = 3,
                 coord_types: List[str] = ['x', 'y', 'r'], bias_length: int = 1,
                 fourier_features: int = 0, default_imsize: Tuple[int, int] = (128,128)) -> None:
        self.channels = colour_channels
        self.coord_types = coord_types
        self.bias_length = bias_length
        self.fourier_features = fourier_features
        self.bias_vec = np.tile(1, bias_length) #np.random.normal(size=bias_length)
        if fourier_features:
            self.fourier_map_vector = fourier.initialize_fourier_mapping_vector(n_features=fourier_features)
        else:
            self.fourier_map_vector = None
        self.perlin_seed = np.random.randint(0, 100)
        # To save recalculating coords each time, keep the coords for the default image size.
        self.default_imsize = default_imsize
        self.default_coords = ImageCreator.get_coords(imsize=self.default_imsize, types=self.coord_types, perlin_seed=self.perlin_seed)  
        self.save_loc = save_loc
        self.base_name = datetime.now().strftime("%d%b%Y_%I%p%M")

    @property
    def n_in(self):
         if self.fourier_features:
            return (self.fourier_features*2) + self.bias_length
         else:
            return len(self.coord_types) + self.bias_length
    
    @property
    def n_out(self):
        return self.channels        
    
    def create_image(self, genome: Genome, imsize: Tuple[int, int] = (128,128)) -> Image:
        if imsize == self.default_imsize:
            coord_dict = self.default_coords
        else:
            coord_dict = ImageCreator.get_coords(imsize=imsize, types=self.coord_types, perlin_seed=self.perlin_seed)
        cppn = NNFF(genome)
        pixels = imsize[0]*imsize[1]
        bias_tile = np.tile(self.bias_vec, (pixels, 1))  # The bias (scalar or vector must go in for every pixel)
        if self.fourier_features:
            # Use fourier features rather than coordinates as inputs to the neural net.
            use_coords = np.stack([coord_dict['x'], coord_dict['y']], axis=-1)
            feats = fourier.fourier_mapping(use_coords, self.fourier_map_vector)
            n_ffeats = self.fourier_map_vector.shape[0]
            feats = feats.reshape(pixels, n_ffeats*2) # times two because we have sin and cos features
            img_raw = np.array(cppn.feedforward((*feats.T, *bias_tile.T)))
        else:
             # Vanilla CPPN with coordinate inutes to neural network
            use_coords = [coord_dict.get(a).ravel() for a in self.coord_types]
            img_raw = np.array(cppn.feedforward((*use_coords, *bias_tile.T)))
        if self.channels == 1:
            img_square = img_raw.T.reshape(imsize[0], imsize[1])
        else:
            img_square = img_raw.T.reshape(imsize[0], imsize[1], self.channels)
        image_out = Image(img_square)
        image_out.genome = genome  # The image has a reference to its own genome
        image_out.creator = self # give it a ref to the creator in case we need to upscale it.
        return image_out
        
        
        # cppn = CPPN(NNFF(genome), self.coord_types, self.bias_vec, self.fourier_map_vector)
        # img = cppn.create_image( (size, size))
        # img.genome = genome  # The image has a reference to its own genome
        # img.creator = self  # give it a ref to the creator in case we need to upscale it.
        # return img
    
    @staticmethod
    def get_coords(imsize: Tuple[int, int] = (128,128), types: List[str] = ['x','y','r','phi','perlin'], perlin_seed=None) -> Dict[str, np.ndarray]:
        all_coords = dict()
        
        # Cartesian coordinates
        x_axis = np.linspace(-1, 1, imsize[0])
        y_axis = np.linspace(-1, 1, imsize[1])
        x_coords, y_coords = np.meshgrid(x_axis, y_axis)
        if 'x' in types:
            all_coords['x'] = x_coords
        if 'y' in types:
            all_coords['y'] = y_coords  
        # Polar coordinates
        if 'r' in types:
            r_coords = np.sqrt(x_coords ** 2 + y_coords ** 2)
            all_coords['r'] = r_coords
        if 'phi' in types:
            phi_coords = np.arctan2(y_coords, x_coords)
            all_coords['phi'] = phi_coords
        # Perlin noise
        if 'perlin' in types:
            perlin = get_perlin_noise(shape=imsize, seed=perlin_seed)
            all_coords['perlin'] = perlin
        return all_coords


class Image:
    """
    Simple class to hold image data and some metadata
    like size and number of channels.
    Might also have save and show methods etc.
    """
    _image_num = 0

    def __init__(self, data : np.ndarray, genome: Genome = None, creator: ImageCreator = None):
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
        self.genome = genome
        self.creator = creator
        self._image_num += 1

    @property
    def name(self) -> str:
        n = 'img_{}'.format(self.genome.metadata.get('name', str(self._image_num)))
        if 'species' in self.genome.metadata:
            n = n + '_' + self.genome.metadata.get('species')
        return n

    def change_resolution(self, resolution: int) -> Image:
        """
        Returns the image at the requested resolution
        """
        if (resolution, resolution) == self.size:
            # We already have the correct resolution
            # We can just return the image we already have
            return self
        else:
            # Create a copy at the requested resolution
            new_img = self.creator.create_image(self.genome, (resolution,resolution))
            return new_img

    def save_img(self, filename: str = None, resolution: int = None) -> str:
        """
        Saves the image.
        """
        if resolution:
            save_data = self.change_resolution(resolution).data
        else:
            save_data = self.data
        savepath = self._get_savepath(filename) + '.png'
        imsave(savepath, save_data, vmin=0, vmax=1, cmap='gray') # Note the cmap param is ignored if we have RGB data.
        print('Image saved here: {}'.format(savepath))
        return savepath

    def save_genome(self, filename: str = None):
        savepath = self._get_savepath(filename) + '.json'
        self.genome.save(savepath)
        print('Genome saved here: {}'.format(savepath))
        return savepath

    def _get_savepath(self, filename: str = None):
        if filename:
            filename, suffix = os.path.splitext(filename)
            savedir = self.creator.save_loc
        else:
            filename = self.name
            savedir = os.path.join(self.creator.save_loc, self.creator.base_name)
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        savepath = os.path.join(savedir, filename)
        return savepath

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
    
    def show(self, ax: plt.Axes = None) -> None:
        """
        Shows the image in the plots window.
        Optionally pass in an Axes to plot on
        (useful if arranging subplots etc.)
        """
        cmap = 'gray' if self.channels==1 else None
        if ax:
            ax.imshow(self.data, cmap=cmap, vmin=0, vmax=1, extent=[-1,1,1,-1])
        else:
            plt.imshow(self.data, cmap=cmap, vmin=0, vmax=1, extent=[-1,1,1,-1])
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
