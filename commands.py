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

a = g.Genome(3, 3, verbose=True)
#g.add_connection()
a.add_node()
v.netviz(a)
net = n.NNFF(a)
net.feedforward([1,1,1])

b = g.Genome(3,3,verbose=True)
v.netviz(b)

ch = a.crossover(b)
v.netviz(ch)

d = g.Genome(g.n_in, g.n_out, init_conns=False, recurrent=g.recurrent, verbose=g.verbose)
dnodes = d.get_node_ids()

p1 = g.Genome.load('stanley_parent1.json')
p2 = g.Genome.load('stanley_parent2.json')
ch = p1.crossover(p2)
v.netviz(ch)