#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The entry point for running the system

Created on Tue Sep 15 19:43:24 2020

@author: riversdale
"""
import yaml

from evaluators import InteractiveEvaluator, PixelPctEvaluator, ImageNetEvaluator
from image_cppn import ImageCreator
from genome import Genome
from population import Population


def main():
    with open('config2.yaml', 'r') as f:  # TODO: allow any config file to be passed in
        CFG = yaml.safe_load(f)

    # set up components
    image_creator = ImageCreator(save_loc=CFG['save_location'], **CFG['image_settings'])  # poor style? Use another builder?
    seed_genome = Genome(image_creator.n_in, image_creator.n_out, settings=CFG['genome_settings'])
    population = Population(seed_genome=seed_genome, **CFG['population_settings'])

    # evaluator
    evaluation = CFG['evaluation'].lower()
    if evaluation == 'interactive':
        evaluator = InteractiveEvaluator(image_creator)
    elif evaluation == 'target':
        from image_cppn import Image  # TODO: is it bad to have import here?
        target = Image.load(CFG['target_img'], CFG['image_settings']['colour_channels'])
        evaluator = PixelPctEvaluator(image_creator, target_img=target, visible=CFG['visible'])
    elif evaluation == 'imagenet':
        evaluator = ImageNetEvaluator(image_creator, fade_factor=0.98, visible=CFG['visible'])  # TODO: fade_factor magic number - could load from config??

    population.run(evaluator=evaluator, generations=CFG['max_generations'])


if __name__ == "__main__":
    main()