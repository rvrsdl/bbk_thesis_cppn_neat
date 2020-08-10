# -*- coding: utf-8 -*-

"""
Contains various fitness evaluators that can be passed to
a Population object.
"""
from typing import List

import numpy as np

from genome import Genome
from nnet import NNFF
from image_cppn import Image, CPPN
import tk_display as td

# typing definitions
Genomes = List[Genome]
Images = List[Image]

class AbstractEvaluator:
    
    def __init__(self, fourier_map_vec=None, bias_vec=None):
        self.gen_num = 0
        self.fourier_map_vec = fourier_map_vec
        self.bias_vec = bias_vec
        self.gameover = False # We can set this to true to break out of caller loop
    
    def run(self, genomes: Genomes):
        pass
    
    def generate_images(self, genomes: Genomes) -> Images:
        """
        Any evaluator will need to generate images from the
        genomes so this method can be shared between them.
        """
        imgs = []
        for g in genomes:
            cppn = CPPN(NNFF(g), self.fourier_map_vec)
            imgs.append( cppn.create_image((128,128), bias=self.bias_vec) )
        return imgs
    
class InteractiveEvaluator(AbstractEvaluator):
    
    def run(self, genomes: Genomes) -> None:
        if td.aborted:
            self.gameover = True
            return
        self.gen_num +=1
        imgs = self.generate_images(genomes)
        grd = td.ImgGrid(imgs, n_imgs=28, nrows=4, title="Generation {}".format(self.gen_num))
        ratings = grd.run()
        # Now set the genome fitnesses according to the ratings
        for g,r in zip(genomes, ratings):
            g.fitness = r

class PixelDiffEvaluator(AbstractEvaluator):
    
    def __init__(self, target_img: Image = None, fourier_map_vec=None, bias_vec=None):
        self.gen_num = 0
        if target_img is None:
            self.target_img = self.get_default_target()
        else:
            self.target_img = target_img
        self.channels = self.target_img.channels
        # Could do something to see how close RGB colours are
        self.max_dist = self.target_img.size[0] * self.target_img.size[0] * 1
        self.gameover = False 
        self.fourier_map_vec = fourier_map_vec
        self.bias_vec = bias_vec
    
    def run(self, genomes: Genomes) -> None:
        self.gen_num += 1
        imgs = self.generate_images(genomes)
        ratings = [self.l2_dist_rating(img) for img in imgs]
        # Now set the genome fitnesses according to the ratings
        for g,r in zip(genomes, ratings):
            g.fitness = r
            
    def l2_dist_rating(self, img: Image) -> float:
        # TODO: could average pixel values in larger
        # squares for each image first. ie. effectively
        # lower the resolution.
        if self.channels == 1:
            # grayscale so we can just do squared diff of brightness for each pixel
            l2_dist = np.sqrt(np.sum((self.target_img.data.ravel() - img.data.ravel())**2))
            rating = 1 - (l2_dist / self.max_dist)
        elif self.channels == 3:
            # TODO 
            pass
        return rating
    
    def get_default_target(self) -> Image:
        ones = np.tile(1,(64,64))
        zeros = np.tile(0,(64,64))
        left = np.concatenate((ones,zeros), axis=0)
        right = np.concatenate((zeros,ones), axis=0)
        out = np.concatenate((left,right), axis=1)
        return Image(out)
        