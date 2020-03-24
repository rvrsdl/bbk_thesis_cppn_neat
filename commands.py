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
g.add_connection()
g.add_node()
a = v.netviz(g)
a