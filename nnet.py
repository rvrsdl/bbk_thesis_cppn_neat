from __future__ import annotations # enables type hints

import numpy as np
from genome import Genome
import funcs


class NNFF(object):
    """
    This is initialised with a genome object and creates a working feedforward neural net.
    """
    genome = None

    def __init__(self, in_genome: Genome):
        assert not in_genome.recurrent, "NNET_FF can only cope with a non-recurrent genome."
        self.genome = in_genome
        self.layers = self.get_layers_ff()
        self.n_layers = len(self.layers)
        self.layer_info = self.get_layer_info()
        #self.biases = self.get_biases()
        
    def get_layer_info(self):
        wgts_dict = {(gene['from'],gene['to']): gene['wgt'] for gene in self.genome.conn_genes if gene['enabled']}
        from_nodes = []
        layer_dicts = []
        for l in range(self.n_layers-1):
            from_nodes.extend(self.layers[l]) # this accumulates (because a connection can be from ANY previous layer)
            to_nodes = self.layers[l+1] # this does not accumulate
            wgts_mat = np.array([[wgts_dict.get((u, v), 0) for u in from_nodes] for v in to_nodes])
            act_func_strs = [g['act_func'] for g in self.genome.node_genes if g['id'] in to_nodes]
            act_funcs = list(map(funcs.get_funcs, act_func_strs))
            act_args = [g.get('act_args', dict()) for g in self.genome.node_genes if g['id'] in to_nodes]
            layer_dicts.append({
                    'from_nodes': from_nodes.copy(), # create static copy
                    'to_nodes': to_nodes,
                    'wgts_mat': wgts_mat,
                    'act_funcs': act_funcs,
                    'act_args':  act_args
                    })
        return layer_dicts 
    
    def get_layers_ff(self):
        """
        Gets the feedforward layers
        Inspired by: https://github.com/CodeReclaimers/neat-python/blob/master/neat/graphs.py
        """
        conns = self.genome.get_connections(True)
        layers = []
        seen = set(self.genome.get_node_ids('input'))
        layers.append(list(sorted(seen)))
        while True:
            # Get nodes that directly connect to what we have seen
            candidates = {b for (a, b) in conns if a in seen and b not in seen}
            # Limit to nodes whose entire input set has already been seen
            next_layer = set()
            for n in candidates:
                in_feed = {a for (a, b) in conns if b == n}
                if all(map(lambda x: x in seen, in_feed)):
                    next_layer.add(n)

            if not next_layer:
                break

            layers.append(list(sorted(next_layer))) # Sort nodes and convert to tuple
            seen = seen.union(next_layer)

        return layers
    
    def feedforward(self, a):
        """
        Because we might be reusing neurons from previous layers we can't do a 
        simple loop through the layer activations.
        Instead we'll keep a dict of node outputs so we can reuse them.
        (ie. not have to recalculate them)
        """
        assert len(a)==len(self.layers[0]), "Input vector must be same size as input layer."
        node_vals = {node: val for (node,val) in zip(self.layers[0], a)} # Initialise the dict with inputs
        for linfo in self.layer_info:
            inp_vec = [node_vals[n] for n in linfo['from_nodes']]
            weights = linfo['wgts_mat']
            act_funcs = linfo['act_funcs']
            act_args = linfo['act_args']
            out_vec = [f(z, **p) for f, z, p in zip(act_funcs, np.dot(weights, inp_vec), act_args)]
            node_vals.update( {node: val for (node,val) in zip(linfo['to_nodes'],out_vec)} )
        try:
            return [node_vals[n] for n in self.genome.get_node_ids('output')]
        except:
            print('Saving failed genome as ./failed.json')
            self.genome.save(filename='failed.json')
            raise
            