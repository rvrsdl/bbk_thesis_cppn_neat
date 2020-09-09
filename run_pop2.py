#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:15:53 2020

@author: riversdale

run_pop2.py
"""
import numpy as np

from population import Population
from evaluators import ImageNetEvaluator, InteractiveEvaluator, PixelDiffEvaluator, PixelPctEvaluator
import fourier
from nnet import NNFF
from image_cppn import CPPN


# Some settings
auto = 2# 0 for manual; 1 for pexeldiff; 2 for imagenet
channels = 3 # 3 for RGB
fourier_feats = 0 # zero if you don't want to use.
bias_len = 3
bias_vec = np.random.normal(size=bias_len)
if fourier_feats:
    in_layer = (fourier_feats*2) + bias_len
    fmv = fourier.initialize_fourier_mapping_vector(n_features=fourier_feats)
else: 
    in_layer = 3 + bias_len
    fmv = None

pop = Population(popsize=36, in_layer=in_layer, out_layer=channels)
if auto==0:
    evaluator = InteractiveEvaluator(fourier_map_vec=fmv, bias_vec=bias_vec)
elif auto==1:
    from image_cppn import Image
    duck = Image.load('duck_bw_128.png', channels=1)
    evaluator = PixelPctEvaluator(target_img=None, bias_vec=bias_vec, fourier_map_vec=fmv, visible=True)
elif auto==2:
    evaluator = ImageNetEvaluator(channels=channels, fade_factor = 0.98, visible=True)
pop.run(evaluator, generations=50)

# After the run we have the sorted best genomes.
# Look at the best one:
g = pop.this_gen[2]
cppn = CPPN(NNFF(g), fmv)
img = cppn.create_image((128,128))
img.show()

# 
# from image_cppn import Image
# tgt = evaluator.get_default_target()
# tgt.diff(shift=-5).show()

# duck.diff(shift=10).show()

