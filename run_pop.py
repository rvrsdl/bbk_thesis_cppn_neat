#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run and show a population
"""
import os

from matplotlib.image import imsave

from population import Population
from nnet import NNFF
from tk_display import ImgGrid
import cppn
import fourier

CHANNELS = 3
P = 28
FOURIER = False
if FOURIER:
    FFEATS = 52
    n_inputs = FFEATS*2+1
    pop = Population(n_inputs, CHANNELS, popsize=P, mutation_rate=0.8, fourier_feats=FFEATS)
else:
    n_inputs = 4
    pop = Population(n_inputs, CHANNELS, popsize=P, mutation_rate=0.8)
cppn.CHANNELS = CHANNELS


# Placeholder, just to make it a bit more complex:
# for i in range(10):
#     pop.update_pop_fitness()
#     pop.breed_next_gen()

#pop_name = "pop{}".format( cppn.get_epoch_str() )
#os.mkdir("./output/{}".format(pop_name))

# Make an iteractive fitness evaluator using ImgGrid
def interactive_fitness(genomes, gen_num):
    imgs = []
    for i in range(P):
        net = NNFF(genomes[i])
        imgs.append( cppn.create_image2(net, (128,128)) )
        #imsave("./output/{}/img{}.png".format(pop_name,i), imgs[i], vmin=0, vmax=1, cmap='binary')
    #img_grid("output/{}/img*.png".format(pop_name))
    grd = ImgGrid(imgs, n_imgs=28, nrows=4, title="Generation {}".format(gen_num))
    ratings = grd.run()
    # NOw set the genome fitnesses according to the ratings
    for i in range(P):
        genomes[i].fitness = ratings[i]

def interactive_fitness_fourier(genomes, gen_num, fourier_map_vec):
    imgs = []
    for i in range(P):
        net = NNFF(genomes[i])
        imgs.append( cppn.create_image_fourier(net, (128,128), pop.fourier_map_vec) )
        #imsave("./output/{}/img{}.png".format(pop_name,i), imgs[i], vmin=0, vmax=1, cmap='binary')
    #img_grid("output/{}/img*.png".format(pop_name))
    grd = ImgGrid(imgs, n_imgs=28, nrows=4, title="Generation {}".format(gen_num))
    ratings = grd.run()
    # NOw set the genome fitnesses according to the ratings
    for i in range(P):
        genomes[i].fitness = ratings[i]

# We pass an evaluation function to the pop.run method   
# This func must update each genome's fitness
if FOURIER:
    pop.run(interactive_fitness_fourier, generations=100)
else:
    pop.run(interactive_fitness, generations=100)
    
