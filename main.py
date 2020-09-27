#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The entry point for running the system

Created on Tue Sep 15 19:43:24 2020

@author: riversdale
"""
import os
import argparse
import yaml
from typing import List

from src.evaluators import InteractiveEvaluator, PixelPctEvaluator, ImageNetEvaluator
from src.imaging import ImageCreator
from src.genome import Genome
from src.population import Population
from src.imaging import Image

def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = args.config
    head, tail = os.path.split(config)
    if not head:
        # If directory was not specified assume the file is in the configurations folder
        # (one above where we are now).
        head = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configurations')) # put '..', before configrations to go up one
        config = os.path.join(head, tail)

    # Load the config file
    with open(config, 'r') as f:
        CFG = yaml.safe_load(f)

    # resolve save location for output images
    save_loc = CFG['save_location']
    save_loc = os.path.abspath(save_loc) # because path in config file is relative.
    if not os.path.exists(save_loc): # If the output folder doesn't exist, create it now.
        os.mkdir(save_loc)

    # set up components
    image_creator = ImageCreator(save_loc=save_loc, **CFG['image_settings'])
    seed_genome = Genome(image_creator.n_in, image_creator.n_out, settings=CFG['genome_settings'])
    population = Population(seed_genome=seed_genome, **CFG['population_settings'])

    # set up the correct evaluator
    evaluation = CFG['evaluation'].lower()
    if evaluation == 'interactive':
        evaluator = InteractiveEvaluator(image_creator)
    elif evaluation == 'target':
        target = Image.load(CFG['target_img'], CFG['image_settings']['colour_channels'])
        evaluator = PixelPctEvaluator(image_creator, target_img=target, visible=CFG['visible'])
    elif evaluation == 'imagenet':
        evaluator = ImageNetEvaluator(image_creator, fade_factor=0.98, visible=CFG['visible'])  # TODO: fade_factor magic number - could load from config??
    evaluator.breed_method = population.breed_method
    evaluator.thresh = population._thresh
    evaluator.show_gens = CFG['max_generations']
    # run the population
    population.run(evaluator=evaluator, generations=CFG['max_generations'])

if __name__ == "__main__":
    main()