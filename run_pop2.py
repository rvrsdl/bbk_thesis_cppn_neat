#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:15:53 2020

@author: riversdale

run_pop2.py
"""
import numpy as np

from population import Population
from evaluators import InteractiveEvaluator, PixelDiffEvaluator
import fourier
from nnet import NNFF
from image_cppn import CPPN


# Some settings
channels = 1 # for RGB
fourier_feats = 0
bias_len = 1
bias_vec = np.random.normal(size=bias_len)
if fourier_feats:
    in_layer = (fourier_feats*2) + bias_len
    fmv = fourier.initialize_fourier_mapping_vector(n_features=fourier_feats)
else: 
    in_layer = 3 + bias_len
    fmv = None

pop = Population(popsize=105, in_layer=in_layer, out_layer=channels)
# evaluator = InteractiveEvaluator(fourier_map_vec = fmv, bias_vec=bias_vec)
from image_cppn import Image
duck = Image.load('duck_bw_128.png', channels=1)
evaluator = PixelDiffEvaluator(target_img=duck,bias_vec=bias_vec, fourier_map_vec=fmv)
pop.run(evaluator, generations=10)

# After the run we have the sorted best genomes.
# Look at the best one:
g = pop.this_gen[2]
cppn = CPPN(NNFF(g), fmv)
img = cppn.create_image((128,128))
img.show()

#
