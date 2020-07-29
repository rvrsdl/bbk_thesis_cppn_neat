# -*- coding: utf-8 -*-

"""
Contains various fitness evaluators that can be passed to
a Population object.
"""
from typing import List

import numpy as np

from genome import Genome
from nnet import NNFF
from cppn import CPPN
from image import Image
from tk_display import ImgGrid

# typing definitions
Genomes = List[Genome]
Images = List[Image]

class AbstractEvaluator:
    
    def run(self, genomes: Genomes):
        pass
    
    def generate_images(self, genomes: Genomes) -> Images:
        """
        Any evaluator will need to generate images from the
        genomes so this method can be shared between them.
        """
        imgs = []
        for g in genomes:
            cppn = CPPN(NNFF(g))
            imgs.append( cppn.create_image2((128,128)) )
        return imgs
    
class InteractiveEvaluator(AbstractEvaluator):
    
    def __init__(self):
        self.gen_num = 0
    
    def run(self, genomes: Genomes) -> None:
        self.gen_num +=1
        imgs = self.generate_images(genomes)
        grd = ImgGrid(imgs, n_imgs=28, nrows=4, title="Generation {}".format(self.gen_num))
        ratings = grd.run()
        # Now set the genome fitnesses according to the ratings
        for g,r in zip(genomes, ratings):
            g.fitness = r

class PixelDiffEvaluator(AbstractEvaluator):
    
    def __init__(self, target_img: Image):
        self.gen_num = 0
        self.target_img = target_img
        self.channels = target_img.channels
        # Could do something to see how close RGB colours are
        self.max_dist = target_img.size[0] * target_img.size[0] * 1
    
    def run(self, genomes: Genomes) -> None:
        self.gen_num += 1
        imgs = self.generate_images(genomes)
        ratings = [self.l2_dist_rating(img) for img in imgs]
        # Now set the genome fitnesses according to the ratings
        for g,r in zip(genomes, ratings):
            g.fitness = r
            
    def l2_dist_rating(self, img: Image) -> float:
        if self.channels == 1:
            # grayscale so we can just do squared diff of brightness for each pixel
            l2_dist = np.sqrt((self.target_img.ravel() - img.ravel())**2)
            rating = 1 - (l2_dist / self.max_dist)
        elif self.channels == 3:
            # TODO 
            pass
        return rating