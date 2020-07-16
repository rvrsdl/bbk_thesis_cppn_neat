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

class Population(object):

    def __init__(self, in_layer, out_layer, popsize=36, mutation_rate=0.5):
        self.n_breed = self.need_to_breed(popsize)
        self.popsize = popsize
        self.in_layer = in_layer
        self.out_layer = out_layer
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.this_gen = self.create_first_gen()
        self.fitness_func = lambda G: random.random() 
        # would be cool to be able to drop in here either 
        # interavtive user selection func or auto fitness func.
        # for now as a placeholder we'll just use a random fitness.
        
    def create_first_gen(self):
        """
        Initialise the population with N members in the first generation.
        Whole population needs to spring from sinle individual so that
        innovation numbers are shared.
        """
        seed = Genome(self.in_layer, self.out_layer)
        generation = [copy.deepcopy(seed) for i in range(self.popsize)]
        # To get a bit of diversity in the initial population we will 
        # give each one an extra node, and randomise the weights.
        for g in generation:
            g.add_node()
            g.randomise_weights()
        return generation
            
    def update_pop_fitness(self):
        for G in self.this_gen:
            G.fitness = self.fitness_func(G)
    
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
        self.this_gen.sort(key= lambda g: g.fitness, reverse=True)
        selected = self.this_gen[:self.n_breed]
        pairings = itertools.combinations(selected,2) # finds all possible combos of 2
        offspring = [p1.crossover(p2, mut_rate=self.mutation_rate) for p1, p2 in pairings]
        self.this_gen = offspring
        self.generation += 1
        print(self)
    
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
    
    def run(self, eval_func, generations=5):
        # eval_func should take a list of genomes and set each of their fitness
        for i in range(generations):
            eval_func(self.this_gen, self.generation)
            self.breed_next_gen()
    
    def __str__(self):
        out = ["Population stats:"]
        out.append("- Generation: {:d}".format(self.generation))
        out.append("- Size: {:d}".format(self.popsize))
        out.append("- Mutation Rate: {}".format(self.mutation_rate))
        n_conns = [len(g.get_conn_ids(only_enabled=True)) for g in self.this_gen]
        n_conn_pctile = [np.percentile(n_conns, p) for p in [25,50,75]]
        out.append("- Complexity: {}, {}, {}".format(*n_conn_pctile))
        return '\n'.join(out)

