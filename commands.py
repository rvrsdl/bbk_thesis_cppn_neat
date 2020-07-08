#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rough commands
Created on Sun Mar 22 13:56:41 2020

@author: riversdale
"""

import nnet as z
import netviz as v

g = z.Genome(3, 3, verbose=True)
#g.add_connection()
g.add_node()
v.netviz(g)

g2 = z.Genome(3,3,verbose=True)
v.netviz(g2)

ch = g.crossover(g2)
v.netviz(ch)

d = z.Genome(g.n_in, g.n_out, init_conns=False, recurrent=g.recurrent, verbose=g.verbose)
dnodes = d.get_node_ids()

p1 = z.Genome.load('stanley_parent1.json')
p2 = z.Genome.load('stanley_parent2.json')
ch = p1.crossover(p2)
v.netviz(ch)