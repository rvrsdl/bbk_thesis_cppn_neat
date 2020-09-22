#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rough commands
Created on Sun Mar 22 13:56:41 2020

@author: riversdale
"""

import genome as g
import visualise as v
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
from visualise import netviz

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
from visualise import netviz
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

# Examining the layers
import numpy as np
from genome import Genome
import cppn
from nnet import NNFF
from visualise import netviz
G = Genome(4,3)
G.mutate()
G.mutate()
netviz(G)
net = NNFF(G)

# ------
# Trying to generate a fe simple illustrative genomes
from image_cppn import ImageCreator
from genome import Genome
import numpy as np

# nofunc, xcoords
G = Genome(4, 1, init_conns=False, output_funcs=['sigmoid'])
G._create_conn_gene(path=(0,4), wgt=2)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show()

# abs, xcoords
G = Genome(4, 1, init_conns=False, output_funcs=['sigmoid'])
G._node_genes.append({
    'id': 5,
    'layer': 'hidden',
    'agg_func': 'sum',
    'act_func': 'abs',
    'act_args': {}
})
G._create_conn_gene(path=(0,5), wgt=1)
G._create_conn_gene(path=(3,4), wgt=-2) #bias
G._create_conn_gene(path=(5,4), wgt=3)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show()

# round, dcoords
G = Genome(4, 1, init_conns=False, output_funcs=['sigmoid'])
G._node_genes.append({
    'id': 5,
    'layer': 'hidden',
    'agg_func': 'sum',
    'act_func': 'round',
    'act_args': {}
})
G._create_conn_gene(path=(2,5), wgt=0.5)
G._create_conn_gene(path=(3,5), wgt=-1) #bias
G._create_conn_gene(path=(5,4), wgt=3)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show()

# sin, ycoors
G = Genome(4, 1, init_conns=False, output_funcs=['sigmoid'])
G._node_genes.append({
    'id': 5,
    'layer': 'hidden',
    'agg_func': 'sum',
    'act_func': 'sin',
    'act_args': {}
})
G._create_conn_gene(path=(1,5), wgt=10)
G._create_conn_gene(path=(3,5), wgt=1) #bias
G._create_conn_gene(path=(5,4), wgt=1.5)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show()

# relu, xcoords
G = Genome(4, 1, init_conns=False, output_funcs=['sigmoid'])
G._node_genes.append({
    'id': 5,
    'layer': 'hidden',
    'agg_func': 'sum',
    'act_func': 'relu',
    'act_args': {}
})
G._create_conn_gene(path=(0,5), wgt=3)
G._create_conn_gene(path=(3,4), wgt=-2) #bias
G._create_conn_gene(path=(5,4), wgt=1)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show()

# mod, xcoords+ycoords
G = Genome(4, 1, init_conns=False, output_funcs=['sigmoid'])
G._node_genes.append({
    'id': 5,
    'layer': 'hidden',
    'agg_func': 'sum',
    'act_func': 'mod',
    'act_args': {}
})
G._create_conn_gene(path=(0,5), wgt=4)
G._create_conn_gene(path=(1,5), wgt=4)
G._create_conn_gene(path=(3,4), wgt=-1) #bias
G._create_conn_gene(path=(5,4), wgt=1)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show()

#---
M3 = np.sum(weights[:, :, None] * inp_vec[None, :, :], axis = 1)
M4 = np.max(weights[:, :, None] * inp_vec[None, :, :], axis = 1)


