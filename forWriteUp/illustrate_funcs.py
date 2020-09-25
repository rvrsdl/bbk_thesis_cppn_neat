#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOR PROJECT WRITE-UP ONLY
Generating a few illustrative plots of activation
functions and what image elements they can produce.

Created on Tue Sep 22 20:58:33 2020

@author: riversdale
"""
import numpy as np
import matplotlib.pyplot as plt

from src.imaging import ImageCreator
from src.genome import Genome
from src import funcs

settings = {'output_funcs': [{'func': 'sigmoid', 'prob': 1}]}
fig, axs = plt.subplots(7, 2,  figsize=[5, 9.3])

# sigmoid xcoords
x = np.linspace(-1, 1, 128)
y = funcs.sigmoid(x * 5)
axs[0,0].plot(x,y)
axs[0,0].set_xlim([-1,1])
axs[0,0].set_ylim([0,1])
axs[0,0].set_title('y = sigmoid(5x)')
axs[0,0].set_ylabel('(a)', y=1, fontweight='bold', labelpad=10, rotation=0)

G = Genome(4, 1, init_conns=False, settings=settings)
G._create_conn_gene(path=(0,4), wgt=5)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show(axs[0,1])
axs[0,1].set_title('x_coords -> Sigmoid')


# abs, xcoords
x = np.linspace(-1, 1, 128)
y = funcs.absz(x)
axs[1,0].plot(x,y)
axs[1,0].set_xlim([-1,1])
axs[1,0].set_ylim([0,1])
axs[1,0].set_title('y = abs(x)')
axs[1,0].set_ylabel('(b)', y=1, fontweight='bold', labelpad=10, rotation=0)


G = Genome(4, 1, init_conns=False, settings=settings)
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
img.show(axs[1,1])
axs[1,1].set_title('x_coords -> Abs')


# round, dcoords
x = np.linspace(-1, 1, 128)
y = funcs.round1dp(x / 2) * 2
axs[2,0].plot(x,y)
axs[2,0].set_xlim([-1,1])
axs[2,0].set_ylim([-1,1])
axs[2,0].set_title('y = round(x)')
axs[2,0].set_ylabel('(c)', y=1, fontweight='bold', labelpad=10, rotation=0)


G = Genome(4, 1, init_conns=False, settings=settings)
G._node_genes.append({
    'id': 5,
    'layer': 'hidden',
    'agg_func': 'sum',
    'act_func': 'round',
    'act_args': {}
})
G._create_conn_gene(path=(2,5), wgt=0.5)
G._create_conn_gene(path=(3,5), wgt=-0.3) #bias
G._create_conn_gene(path=(5,4), wgt=3)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show(axs[2,1])
axs[2,1].set_title('d_coords -> Round')


# sin, ycoors
x = np.linspace(-1, 1, 128)
y = funcs.sinz(10 * x)
axs[3,0].plot(x,y)
axs[3,0].set_xlim([-1,1])
axs[3,0].set_ylim([-1,1])
axs[3,0].set_title('y = sin(10x)')
axs[3,0].set_ylabel('(d)', y=1, fontweight='bold', labelpad=10, rotation=0)

G = Genome(4, 1, init_conns=False, settings=settings)
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
img.show(axs[3,1])
axs[3,1].set_title('y_coords -> Round')

# relu, xcoords
x = np.linspace(-1, 1, 128)
y = funcs.relu(x)
axs[4,0].plot(x,y)
axs[4,0].set_xlim([-1,1])
axs[4,0].set_ylim([0,1])
axs[4,0].set_title('y = relu(x)')
axs[4,0].set_ylabel('(e)', y=1, fontweight='bold', labelpad=10, rotation=0)

G = Genome(4, 1, init_conns=False, settings=settings)
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
img.show(axs[4,1])
axs[4,1].set_title('x_coords -> ReLU')


# mod, xcoords+ycoords
x = np.linspace(-1, 1, 128)
y = funcs.modz(x * 2, thresh=0.1)
axs[5,0].plot(x,y)
axs[5,0].set_xlim([-1,1])
axs[5,0].set_ylim([0,1])
axs[5,0].set_title('y = mod(2x)<0.1')
axs[5,0].set_ylabel('(f)', y=1, fontweight='bold', labelpad=10, rotation=0)


G = Genome(4, 1, init_conns=False, settings=settings)
G._node_genes.append({
    'id': 5,
    'layer': 'hidden',
    'agg_func': 'sum',
    'act_func': 'mod',
    'act_args': {'thresh':0.1}
})
G._create_conn_gene(path=(0,5), wgt=2)
G._create_conn_gene(path=(1,5), wgt=2)
G._create_conn_gene(path=(3,4), wgt=-1) #bias
G._create_conn_gene(path=(5,4), wgt=7)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show(axs[5,1])
axs[5,1].set_title('x_coords, y_coords -> Mod')


# point, xcoords
x = np.linspace(-1, 1, 128)
y = funcs.point(x, p=-0.5)
axs[6,0].plot(x,y)
axs[6,0].set_xlim([-1,1])
axs[6,0].set_ylim([0,1])
axs[6,0].set_title('y = abs(x-0.5)<0.05')
axs[6,0].set_ylabel('(g)', y=1, fontweight='bold', labelpad=10, rotation=0)

G = Genome(4, 1, init_conns=False, settings=settings)
G._node_genes.append({
    'id': 5,
    'layer': 'hidden',
    'agg_func': 'sum',
    'act_func': 'point',
    'act_args': {'p': -0.5}
})
#G._create_conn_gene(path=(2,5), wgt=1)
G._create_conn_gene(path=(0,5), wgt=1)
G._create_conn_gene(path=(3,4), wgt=-1) #bias
G._create_conn_gene(path=(5,4), wgt=7)
C = ImageCreator(colour_channels=1)
C.bias_vec = [1]
img = C.create_image(G)
img.show(axs[6,1])
axs[6,1].set_title('x_coords -> Point')

fig.tight_layout()
plt.show()

# # trying to get floating square off centre!!
# G = Genome(4, 1, init_conns=False, settings=settings)
# G._node_genes.append({
#     'id': 5,
#     'layer': 'hidden',
#     'agg_func': 'min',
#     'act_func': 'nofunc',
#     'act_args': {}
# })
# G._node_genes.append({
#     'id': 6,
#     'layer': 'hidden',
#     'agg_func': 'max',
#     'act_func': 'nofunc',
#     'act_args': {}
# })
# G._node_genes.append({
#     'id': 7,
#     'layer': 'hidden',
#     'agg_func': 'sum',
#     'act_func': 'nofunc',
#     'act_args': {}
# })
# G._node_genes.append({
#     'id': 8,
#     'layer': 'hidden',
#     'agg_func': 'sum',
#     'act_func': 'nofunc',
#     'act_args': {}
# })
# # min of x & y
# G._create_conn_gene(path=(0,5), wgt=1)
# G._create_conn_gene(path=(1,5), wgt=1)
# # is greater than 0.5
# G._create_conn_gene(path=(5,7), wgt=1)
# G._create_conn_gene(path=(3,7), wgt=-0.5)
# # max of x & y
# G._create_conn_gene(path=(0,6), wgt=1)
# G._create_conn_gene(path=(1,6), wgt=1)
# # is less than 0.5
# G._create_conn_gene(path=(6,8), wgt=1)
# G._create_conn_gene(path=(3,8), wgt=0.5)
# # output
# G._create_conn_gene(path=(7,4), wgt=1)
# G._create_conn_gene(path=(8,4), wgt=1)
# # image
# C = ImageCreator(colour_channels=1)
# C.bias_vec = [1]
# img = C.create_image(G)
# img.show()