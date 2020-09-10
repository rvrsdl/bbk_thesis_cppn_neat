#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:32:47 2020

@author: riversdale
"""
import copy

import numpy as np

class HallOfFame(object):
    """
    An object to "save" the best individuals of a run.
    Basically just a dictionary
    """
    
    def __init__(self, max_per_species=1):
        self.members = dict()
        self.max_per_species = max_per_species
        
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
            species = candidate.metadata.get('species')
            if species:
                if species in self.members:
                    # Check whether the candidate makes the cut.
                    existing = self.members[species]
                    if len(existing)<self.max_per_species:
                        self.members[species].append(prepare(candidate))
                    else:
                        exist_fit = [e.metadata.get('fitness') for e in existing]
                        if candidate.metadata.get('fitness') > min(exist_fit):
                            # Replace the minimum fitness existing member with the candidate
                            existing[np.argmin(exist_fit)] = prepare(candidate)
                            self.members[species] = existing
                else:
                    # If the species isn't yet in the hall of fame then add it
                    self.members[species] = [prepare(candidate)]
            else:
                # If the candidate doesn't have a species there's nothing we can do
                print('Not saving genome without species.')
                
        

        