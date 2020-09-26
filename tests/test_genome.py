# Testing the Genome class
import os
from src.genome import Genome
#from ..src.visualise import render_saved_genome

#render_saved_genome('stanley_parent1.json')

def test_init():
    """
    Test that the __init__ method of a genome sets up the correct
    number of input and output nodes. (connections between them
    are initialised randomly).
    """
    genome = Genome(4, 5)
    assert len(genome.get_node_ids(layer='input')) == 4
    assert len(genome.get_node_ids(layer='output')) == 5

def test_crossover():
    """
    Tests the crossover operation on two genomes.
    Uses the genomes described in figure 4. of the original
    Stanley & Miikkulainen paper "Evolving Neural Networks through Augmenting Topologies"
    which explained how the crossover operation should work.
    """
    # Load the Genomes used in the paper (manually encoded and saved by me)
    parent1 = Genome.load('./tests/stanley_parent1.json')
    parent2 = Genome.load('./tests/stanley_parent2.json')
    parent1._settings['disable_prob'] = 1 # To remove randomness from the crossover operation.
    offspring = parent1.crossover(parent2)
    expected_connections = {(0,3), (0,4), (0,5), (1,4), (2,3), (2,4), (4,5), (5,3)}
    assert offspring.get_connections() == expected_connections

def test_add_node():
    """
    Tests whether the add_node operation (one of the mutation types) works.
    """
    genome = Genome(3,4) # initialise a basic genome
    orig_nodes = genome.get_node_ids(layer='all')
    genome.add_node()
    new_nodes = genome.get_node_ids(layer='all')
    # Check that a node has been added
    assert len(new_nodes) - len(orig_nodes) == 1

def test_add_connection():
    """
    Tests whether the add_connection() operation (one of the mutation types) works.
    """
    genome = Genome(3, 4)  # initialise a basic genome
    orig_conns = genome.get_connections(only_enabled=False)
    genome.add_connection()
    new_conns = genome.get_connections(only_enabled=False)
    assert len(new_conns) - len(orig_conns) == 1