#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:03:29 2020

@author: riversdale
"""
import argparse
import os
from typing import List

import graphviz as gv
from genome import Genome

def netviz(genome, show_wgts=False, inp_desc: List[str] = None):
    """
    Pass in a genome - prints a quick visualisation of the network.
    """
    if not inp_desc:
        inp_desc = ['x_coords', 'y_coords', 'd_coords']
        inp_desc.extend(['bias'] * (genome.n_in-3))
    if genome.n_out == 1:
        out_desc = ['Grayscale']
    elif genome.n_out == 3:
        out_desc = ['Red', 'Green', 'Blue']
    else:
        out_desc = ['unknown']*genome.n_out
    all_desc = inp_desc + out_desc

    nn = gv.Digraph()
    for i in genome.get_node_ids():
        node_info = genome.get_node_gene(i)
        if node_info['layer'] == 'input':
            desc = all_desc[i]
            label = "{id}: {layer}\n{desc}".format(desc=desc, **node_info)
            colour = 'blue'
        elif node_info['layer'] == 'output':
            desc = all_desc[i]
            label = "{id}: {layer}\nact: {act_func}\nagg: {agg_func}\n{desc}".format(desc=desc, **node_info)
            colour = 'red'
        else:
            label = "{id}: {layer}\nact: {act_func}\nagg: {agg_func}".format(**node_info)
            colour = 'black'
        nn.node(str(i), label=label, color=colour)

    # for i in genome.get_node_ids('input'):
    #     nn.node(str(i), label='Inp'+str(i), color='blue')
    # for i in genome.get_node_ids('hidden'):
    #     nn.node(str(i), label='Hid'+str(i), color='black')
    # for i in genome.get_node_ids('output'):
    #     nn.node(str(i), label='Out'+str(i), color='red')
    
    for k, v in genome.get_wgts_dict().items():
        if show_wgts:
            label = "%0.2f" % v
        else:
            label = ''
        nn.edge(str(k[0]), str(k[1]), label=label)
    return nn


def render_saved_genome(filepath, inp_desc: List[str] = None):
    filename, suffix = os.path.splitext(filepath)
    assert suffix == '.json', "File must be a JSON file"
    G = Genome.load(filepath)
    N = netviz(G, show_wgts=True, inp_desc=inp_desc)
    N.format = 'svg'
    out_fn = filename + '.svg'
    N.render(filename=out_fn, view=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename
    head, tail = os.path.split(filename)
    if not head:
        # If directory not specified assume file is in current working directory
        filename = os.path.join(os.getcwd(), tail)
    render_saved_genome(filename)


if __name__ == "__main__":
    main()

# /home/riversdale/Documents/MSc/thesis/rw_neat/output/22Sep2020_04PM17/img_gen2_20.json