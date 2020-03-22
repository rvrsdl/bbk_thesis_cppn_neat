#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:03:29 2020

@author: riversdale
"""


import graphviz as gv

def netviz(genome, show_wgts=False):
    """
    Pass in a genome - prints a quick visualisation of the network.
    """
    nn = gv.Digraph()
    for i in genome.get_node_ids('input'):
        nn.node(str(i), 'Inp'+str(i), color='blue')
    for i in genome.get_node_ids('hidden'):    
        nn.node(str(i), 'Hid'+str(i), color='black')
    for i in genome.get_node_ids('output'):    
        nn.node(str(i), 'Out'+str(i), color='red')
    
    for conn in genome.conn_genes:
        if conn['enabled']:
            if show_wgts:
                label = "%0.2f" % conn['wgt']
            else:
                label = ''
            nn.edge(str(conn['from']), str(conn['to']), label=label)
            
    return nn