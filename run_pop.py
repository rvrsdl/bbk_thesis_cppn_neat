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

CHANNELS = 3
cppn.CHANNELS = CHANNELS
pop = Population(4, CHANNELS)

# Placeholder, just to make it a bit more complex:
# for i in range(10):
#     pop.update_pop_fitness()
#     pop.breed_next_gen()

#pop_name = "pop{}".format( cppn.get_epoch_str() )
#os.mkdir("./output/{}".format(pop_name))

def interactive_fitness(genomes, gen_num):
    imgs = []
    for i in range(35):
        net = NNFF(genomes[i])
        imgs.append( cppn.create_image2(net, (128,128)) )
        #imsave("./output/{}/img{}.png".format(pop_name,i), imgs[i], vmin=0, vmax=1, cmap='binary')
    #img_grid("output/{}/img*.png".format(pop_name))
    grd = ImgGrid(imgs, title="Generation {}".format(gen_num))
    ratings = grd.run()
    # NOw set the genome fitnesses according to the ratings
    for i in range(35):
        genomes[i].fitness = ratings[i]
    
# We pass an evaluation function to the pop.run method   
# This func must update each genome's fitness
pop.run(interactive_fitness)

