#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:32:47 2020

@author: riversdale
"""
import copy

import numpy as np

from src.nnet import NNFF
from src.image_cppn import CPPN

class HallOfFame(object):
    """
    An object to "save" the best individuals of a run.
    Basically just a dictionary
    """
    
    def __init__(self, max_per_species=1, min_fitness=75):
        self.members = dict()
        self.max_per_species = max_per_species
        self.min_fitness = min_fitness
        
    def cache(self, pop):
        """
        Goes through a population and caches the best of each species.
        """
        def prepare(g):
            """
            A helper func to copy the genome and add which generation it
            appeared in to its metadata.
            """
            g_copy = copy.deepcopy(g)
            g_copy.metadata['generation'] = pop.generation
            return g_copy
        
        for candidate in pop.this_gen:
            cand_fit = candidate.get_fitness(raw=True)
            if cand_fit >= self.min_fitness:
                species = candidate.metadata.get('species')
                if species:
                    if species in self.members:
                        # Check whether the candidate makes the cut.
                        existing = self.members[species]
                        if len(existing)<self.max_per_species:
                            self.members[species].append(prepare(candidate))
                        else:
                            exist_fit = [e.get_fitness(raw=True) for e in existing]
                            if cand_fit > min(exist_fit):
                                # Replace the minimum fitness existing member with the candidate
                                existing[np.argmin(exist_fit)] = prepare(candidate)
                                self.members[species] = existing
                    else:
                        # If the species isn't yet in the hall of fame then add it
                        self.members[species] = [prepare(candidate)]
                else:
                    # If the candidate doesn't have a species there's nothing we can do
                    print('Not saving genome without species.')
                
    def show(self, per_species=1):
        """
        Show the best image in each category
        """
        imgs = []
        text = []
        for k,v in self.members.items():
            v.sort(key= lambda g: g.get_fitness(raw=True), reverse=True)
            for i in range(per_species):
                cppn = CPPN(NNFF(v[i]), fmv)
                imgs.append( cppn.create_image((128,128), bias=bias_vec) )
