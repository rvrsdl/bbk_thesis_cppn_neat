"""
To handle the population dynamics.
This will be the entry point for the evolutionary algorithm.
"""
from math import factorial
import itertools
import copy
import random

import numpy as np

from genome import Genome
import evaluators

class Population(object):

    def __init__(self, in_layer, out_layer, popsize=36, mutation_rate=0.5, hall_of_fame=None):
        self.n_breed = self.need_to_breed(popsize)
        self.popsize = popsize
        self.in_layer = in_layer
        self.out_layer = out_layer
        self.mutation_rate = mutation_rate
        self.generation = 0 
        self.this_gen = None
        self.hall_of_fame = hall_of_fame
        self.breed_method = 'selected' #, 'total', 'species'

    def create_first_gen(self):
        """
        Initialise the population with N members in the first generation.
        Whole population needs to spring from sinle individual so that
        innovation numbers are shared.
        """
        seed = Genome(self.in_layer, self.out_layer)
        first_gen = [copy.deepcopy(seed) for i in range(self.popsize)]
        # To get a bit of diversity in the initial population we will 
        # give each one an extra node, and randomise the weights.
        for g in first_gen:
            g.add_node()
            g.add_connection()
            g.randomise_weights()
        self.this_gen = first_gen
        self.generation = 1
    
    def breed_next_gen(self):
        """
        The most fit N individuals in the population breed with
        each other (all possible pairwise combinations), where N
        is a number which causes the pairwise combinations to be
        equal to the desired population size. This has already
        been calculated as self.n_breed.
        
        Offspring are mutated according to the mutation rate
        (should this be set at the pop level or the genome level?)
        TODO: implement species so that breeding only takes place
        between members of the same species.
        """
        if self.breed_method == 'species':
            species = self.species_divide()
            groups = list(species.values())
            n_species = len(species)
            mixed_alloc = round(self.mixed_pct * self.popsize)
            pure_alloc = self.popsize - mixed_alloc
            raw_spec_alloc = pure_alloc / n_species
            n_ceil = self.popsize - (np.floor(raw_spec_alloc) * n_species)
            n_floor = n_species - n_ceil
            group_allocs = np.concatenate([np.tile(np.ceil(raw_spec_alloc), n_ceil), np.tile(np.floor(raw_spec_alloc), n_floor)])
            for grp, n in zip(groups, group_allocs):
                print('TODO')
        elif self.breed_method == 'total':
            # Breeding as many as we need to totally renew the population
            self.this_gen.sort(key= lambda g: g.get_fitness(), reverse=True)
            selected = self.this_gen[:self.n_breed]
            pairings = itertools.combinations(selected,2) # finds all possible combos of 2
            offspring = [p1.crossover(p2, mut_rate=self.mutation_rate) for p1, p2 in pairings]
        elif self.breed_method == 'selected':
            # Breed only from the selected individuals (fitness>10)
            selected = [g for g in self.this_gen if g.get_fitness()>=10]
            selected.sort(key= lambda g: g.get_fitness(), reverse=True)
            if len(selected)<2:
                print('WARNING: You must select at least two individuals')
                print('Please try again')
                return None
                # ie. keep this_gen the same and don't increase the
                # generation numbers, so the user can try again.
            pairings = list(itertools.combinations(selected,2)) # finds all possible combos of 2
            offspring = selected # keep the chosen ones
            for i in range(self.popsize - len(selected)):
                idx = i % len(pairings) # so we can go round again if we need to
                p1 = pairings[idx][0]
                p2 = pairings[idx][1]
                offspring.append( p1.crossover(p2, mut_rate=self.mutation_rate) )
                
        # Set this_gen to the offspring and increment the generation number. 
        self.this_gen = offspring
        self.generation += 1
        print(self)
    
    def species_divide(self):
        """
        Puts members of the current population in a dictionary
        whose keys are the species names.
        Genomes with no species go into the "?" group.
        """
        [(pair, pair[0]*pair[1]) for pair in itertools.combinations(range(1,6),2)]
        k = list(itertools.combinations(range(1,6),2))
        k.sort(key = lambda p: max(p)+min(p)/100)
        species_dict = dict()
        for g in self.this_gen:
            s = g.metadata.get('species','?')
            if s in species_dict:
                species_dict[s].append(g)
            else:
                species_dict[s] = [g]
        # now sort by fitness within species
        for s in species_dict.values():
            s.sort(key= lambda g: g.get_fitness(), reverse=True)
        return species_dict
    
    
    def need_to_breed(self, popsize):
        """
        Returns the minimum number of individuals who need to breed
        (all pairwaise combinations of those individuals) to create
        the required population size in the next generation.
        If this is not a whole number then the proposed population
        size is deemed invalid.
        Formula derived frm the combination formula:
            n is number of individuals
            r is how many are taken at a time (always 2 for us)
            C(n,r) = n!/((n-r)!r!)
        We calculate C (generated population) for different values
        of n (number of individuals who breed) from 3 to 30
        (corresponding to population sizes from 3 to 435)
        TODO: This will be complicated when we introduce speciation
        as not all individuals will be allowed to breed.
        """
        options = np.array([(n, factorial(n)/(factorial(n-2)*2) ) for n in range(3,31)])
        assert popsize in options[:,1], "Invalid population size specified"
        return int(options[options[:,1]==popsize, 0])

    
    def identify_species(self):
        pass
    
    def run(self, evaluator : evaluators.AbstractEvaluator, generations : int = 5):
        # eval_func should take a list of genomes and set each of their fitness
        for i in range(generations):
            if evaluator.gameover:
                print('Ending run early...')
                break
            else:
                if self.generation==0:
                    self.create_first_gen()
                else:
                    self.breed_next_gen()
                evaluator.run(self.this_gen, self.generation)
                if self.hall_of_fame:
                    self.hall_of_fame.cache(self) # put any outstanding members of theis generation in the hall of fame.
        self.this_gen.sort(key= lambda g: g.get_fitness(), reverse=True)
         
    def __str__(self):
        out = ["Population stats:"]
        out.append("- Generation: {:d}".format(self.generation))
        out.append("- Size: {:d}".format(self.popsize))
        out.append("- Mutation Rate: {}".format(self.mutation_rate))
        n_conns = [len(g.get_conn_ids(only_enabled=True)) for g in self.this_gen]
        #n_conn_pctile = [np.percentile(n_conns, p) for p in [25,50,75]]
        n_conn_pctile = [np.min(n_conns), np.median(n_conns), np.max(n_conns)]
        out.append("- Complexity: {}, {}, {}".format(*n_conn_pctile))
        return '\n'.join(out)

