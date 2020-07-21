#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:09:46 2020

@author: riversdale
"""

# Tetsing mod
np.sum((np.mod(np.random.randn(100),0.5)<0.25).astype(int))

# What is likely to be the weighted sum at any node?
# Assume all previous node outptus are in the range -1, 1
# And weights are selected from random normal.
# So:

inp = np.random.uniform(low=-1, high=1, size=5)
wgt = np.random.randn(5)
np.dot(inp, wgt)