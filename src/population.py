"""
To handle the population dynamics.
This will be the entry point for the evolutionary algorithm.
"""
from math import factorial
import itertools
import copy

import numpy as np

from src.genome import Genome
#from caching import HallOfFame
from src.evaluators import AbstractEvaluator


class Population(object):

    def __init__(self, seed_genome: Genome,
                 hall_of_fame = None, **kwargs: int) -> None:
        self._seed = seed_genome
        self.hall_of_fame = hall_of_fame
        self.popsize = kwargs.get('size', 28)
        self.mutation_rate = kwargs.get('mutation_rate', 0.5)
        self.breed_method = kwargs.get('breed_method', 'total')
        if self.breed_method == 'total':
            # Here _thresh refers to the number of genomes which will breed.
            # The top rated {_thresh} genomes will breed, whatever their rating.
            self._thresh = self._need_to_breed(self.popsize)
        elif self.breed_method == 'selected':
            # Here _thresh refers to the minimum score needed to breed
            # Only genomes with a rating > {_thresh} will breed.
            self._thresh = 10
        self.generation = 0
        self.this_gen = None

    def _create_first_gen(self) -> None:
        """
        Initialise the population with N members in the first generation.
        Whole population needs to spring from sinle individual so that
        innovation numbers are shared.
        """
        self.generation = 1
        first_gen = [copy.deepcopy(self._seed) for i in range(self.popsize)]
        # To get a bit of diversity in the initial population we will
        # give each one an extra node, and randomise the weights.
        i=1
        for g in first_gen:
            g.add_node()
            g.add_connection()
            g.randomise_weights()
            g.metadata['name'] = 'gen{}_{}'.format(self.generation, i)
            i += 1
        self.this_gen = first_gen


    def _breed_next_gen(self) -> None:
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
        self.generation += 1
        if self.breed_method == 'species':
            # TODO: complete this.
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
            self.this_gen.sort(key=lambda g: g.get_fitness(), reverse=True)
            selected = self.this_gen[:self._thresh]
            pairings = itertools.combinations(selected, 2)  # finds all possible combos of 2
            offspring = []
            i = 1
            for p1, p2 in pairings:
                child = p1.crossover(p2, mut_rate=self.mutation_rate)
                child.metadata['name'] = 'gen{}_{}'.format(self.generation, i)
                i += 1
                offspring.append(child)
        elif self.breed_method == 'selected':
            # Breed only from the selected individuals (fitness>10)
            selected = [g for g in self.this_gen if g.get_fitness() >= self._thresh]
            selected.sort(key=lambda g: g.get_fitness(), reverse=True)
            if len(selected) < 2:
                print('WARNING: You must select at least two individuals')
                print('Please try again')
                self.generation -= 1  # Undo the increment to the generation number
                return None # We haven't updated this_gen so the user just tries again.
            pairings = list(itertools.combinations(selected, 2))  # finds all possible combos of 2
            offspring = selected # keep the chosen ones
            for i in range(self.popsize - len(selected)):
                idx = i % len(pairings)  # so we can go round again if we need to
                p1 = pairings[idx][0]
                p2 = pairings[idx][1]
                child = p1.crossover(p2, mut_rate=self.mutation_rate)
                child.metadata['name'] = 'gen{}_{}'.format(self.generation, i)
                offspring.append(child)
                
        # Set this_gen to the offspring and increment the generation number. 
        self.this_gen = offspring
        print(self)
    
    def _species_divide(self) -> dict:
        """
        Puts members of the current population in a dictionary
        whose keys are the species names.
        Genomes with no species go into the "?" group.
        """
        # TODO: finish this
        [(pair, pair[0]*pair[1]) for pair in itertools.combinations(range(1, 6), 2)]
        k = list(itertools.combinations(range(1, 6), 2))
        k.sort(key=lambda p: max(p)+min(p)/100)
        species_dict = dict()
        for g in self.this_gen:
            s = g.metadata.get('species', '?')
            if s in species_dict:
                species_dict[s].append(g)
            else:
                species_dict[s] = [g]
        # now sort by fitness within species
        for s in species_dict.values():
            s.sort(key=lambda h: h.get_fitness(), reverse=True)
        return species_dict

    @staticmethod
    def _need_to_breed(popsize):
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
        options = np.array([(n, factorial(n)/(factorial(n-2)*2)) for n in range(3, 31)])
        assert popsize in options[:, 1], "Invalid population size specified"
        return int(options[options[:, 1] == popsize, 0])
    
    def run(self, evaluator: AbstractEvaluator, generations: int = 5):
        # eval_func should take a list of genomes and set each of their fitness
        for i in range(generations):
            if evaluator.gameover:
                print('Ending run early...')
                break
            else:
                if self.generation == 0:
                    self._create_first_gen()
                else:
                    self._breed_next_gen()
                is_final_gen = self.generation==generations
                evaluator.run(self.this_gen, self.generation)
                if self.hall_of_fame:
                    self.hall_of_fame.cache(self) # put any outstanding members of this generation in the hall of fame.
        self.this_gen.sort(key=lambda g: g.get_fitness(), reverse=True)
         
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

