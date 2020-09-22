#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:01:08 2020

@author: riversdale
"""
import yaml

from evaluators import InteractiveEvaluator, PixelPctEvaluator, ImageNetEvaluator
from image_cppn import ImageCreator
from genome import Genome
from population import Population

#client code
def main():
    with open('config.yaml', 'r') as f:  # TODO: allow any config file to be passed in
        CFG = yaml.safe_load(f)
        
    # set up components
    image_creator = ImageCreator(**CFG['image_settings']) # poor style? Use another builder? 
    seed_genome = Genome(image_creator.n_in, image_creator.n_out, **CFG['genome_settings'])
    population = Population(seed_genome=seed_genome, **CFG['population_settings'])

    # evaluator
    evaluation = CFG['evaluation'].lower()
    if evaluation == 'interactive':
        evaluator = InteractiveEvaluator(image_creator)
    elif evaluation == 'target':
        from image_cppn import Image # TODO: is it bad to have import here?
        target = Image.load(CFG['target_img'], CFG['image_settings']['colour_channels'])
        evaluator = PixelPctEvaluator(image_creator, target_img=target, visible=CFG['visible'])
    elif evaluation == 'imagenet':
        evaluator = ImageNetEvaluator(image_creator, fade_factor=0.98, visible=CFG['visible']) #TODO: fade_factor magic number - could load from config??

    population.run(evaluator=evaluator, generations=CFG['max_generations'])

    
    
    # seed genome
    gb = GenomeBuilder()
    gb.set_size(image_creator.n_in, image_creator.n_out) # work out the size needed from the image settings
    gb.set_functions(CFG['functions'], CFG['output_functions'])
    gb.set_mutation_types(CFG['mutation_types'])
    seed_genome = gb.product

    # population (final thing)
    pb = PopulationBuilder()
    pb.set_pop_size(CFG['population_size'])
    pb.set_mutation_rate(CFG['mutation_rate'])
    pb.set_breed_method(CFG['breed_method'])
    pb.set_seed_genome(seed_genome)
    pb.set_evaluator(evaluator) # TODO: pop doesn't yet have this property
    population = pb.product
    
    # off we go!
    population.run()

class GenomeBuilder(object):
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self._genome = Genome()
        
    @property
    def product(self) -> Genome:
        self._genome.init_conns() # actually build the genome
        product = self._genome
        self.reset()
        return product
        
    def set_size(self, in_layer: int, out_layer: int) -> None:
        self._genome.n_in = in_layer
        self._genome.n_out = out_layer
        
    def set_functions(self, funcs: list, output_funcs: list) -> None:
        self._genome.activation_funcs = funcs
        self._genome.output_funcs = output_funcs
        
    def set_mutation_types(self, mutations: dict) -> None:
        self._genome.mutation_types = mutations
    
        
    
class PopulationBuilder(object):
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        self._population = Population()
        
    @property
    def product(self) -> Population:
        self._population.create_first_gen()
        product = self._population
        self.reset()
        return product
        
    def set_pop_size(self, pop_size):
        self._popultion.pop_size = pop_size
        
    def set_mutation_rate(self, mut_rate: float) -> None:
        self._population.mutation_rate = mut_rate
    
    def set_breed_method(self, breed_method: str) -> None:
        self._population.breed_method = breed_method
        
    def set_seed_genome(self, seed: Genome) -> None:
        self._population.seed_genome = seed
        

class Mediator(object):
    
    def __init__(self, population, evaluator, image_creator, ui):
        self.population = population
        self.population.mediator = self
        self.image_creator = image_creator
        self.image_creator.mediator = self
        self.evaluator = evaluator
        self.evaluator.mediator = self
        self.ui = ui
        self.ui.mediator = self
        
    def notify(self, sender : object, event: str, args: dict):
        if sender == self.population:
            if event == 'evaluate':
                print('blah')
                

if __name__ == "__main__":
        main()
        