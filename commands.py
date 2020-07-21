#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rough commands
Created on Sun Mar 22 13:56:41 2020

@author: riversdale
"""

import genome as g
import netviz as v
import nnet as n
import population as p

# a = g.Genome(3, 3, verbose=True)
# #g.add_connection()
# a.add_node()
# v.netviz(a)
# net = n.NNFF(a)
# net.feedforward([1,1,1])

# b = g.Genome(3,3,verbose=True)
# v.netviz(b)

# ch = a.crossover(b)
# v.netviz(ch)

# d = g.Genome(g.n_in, g.n_out, init_conns=False, recurrent=g.recurrent, verbose=g.verbose)
# dnodes = d.get_node_ids()

# p1 = g.Genome.load('stanley_parent1.json')
# p2 = g.Genome.load('stanley_parent2.json')
# ch = p1.crossover(p2)
# v.netviz(ch)

from population import Population
from genome import Genome
from nnet import NNFF
from netviz import netviz

pop = Population(4, 3)
ch = pop.this_gen[1].crossover(pop.this_gen[3])
netviz(pop.this_gen[1])
netviz(pop.this_gen[3])
netviz(ch)

for i in range(10):
    pop.update_pop_fitness()
    pop.breed_next_gen()

net = NNFF(pop.this_gen[4])
img = create_image2(net,(64,64))
show_image(img)
animate(net)

import os
import cppn

cppn.CHANNELS=1
cppn.do_run(num=35)
import tk_display
#grd = tk_display.ImgGrid('output/Thu16July/*128.png', n_imgs=28, nrows=4, ncols=7)
grd = tk_display.ImgGrid('output/e15952*_128.png', n_imgs=28, nrows=4, ncols=7)
grd.run()

G = Genome.load("output/e159499405049.json")
net = NNFF(G)
img = create_image2(net,(128,128))
show_image(img)

# debugging failed genomes
from genome import Genome
from nnet import NNFF
import cppn
from netviz import netviz
G = Genome.load("failed.json")
net = NNFF(G)
img = cppn.create_image2(net, imsize=(64,64))
netviz(G)

# trying with longer noise vector
import numpy as np
import cppn
from nnet import NNFF
noise_len = 4
G = cppn.create_genome(input_nodes=3+noise_len)
net = NNFF(G)
noise = np.random.normal(size=noise_len)
img = cppn.create_image3(net, imsize=(128,128), bias=noise)
cppn.show_image(img)
