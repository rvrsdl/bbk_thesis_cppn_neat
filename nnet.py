from __future__ import annotations # enables type hints
import random
import datetime
import json

import numpy as np


class Genome(object):
    """
    This object manages the genome and handles mutation.
    """

    def __init__(self, n_in, n_out, recurrent=False, verbose=False, init_conns=True):
        """
        Initialise the genes for the basic network in which each output node is connected
        to a random selection of between one and all the input nodes.
        :param n_in: Number of input nodes (including bias)
        :param n_out: Number of output nodes.
        :param recurrent: Whether the network is allowed to be recurrent.
        :param verbose: For debugging.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.recurrent = recurrent
        self.verbose = verbose
        self.innov = 0 # to keep track of the innovation number
        # Add the input and output nodes to the list of node_genes
        self.node_genes = []
        for i in range(n_in):
            self.node_genes.append({
                'id': i,
                'layer': 'input',
                'agg_func': None,
                'act_func': None,
                'bias': None
            })
        for i in range(n_in, n_in+n_out):
            self.node_genes.append({
                'id': i,
                'layer': 'output',
                'agg_func': 'sum',
                'act_func': 'sigmoid',
                'bias': np.random.randn() # From the normal distribution
            })
        self.conn_genes = []
        if init_conns:
            self.init_conns()
        
    def init_conns(self):
        # Now make some connections between inputs and outputs according to the probabilities in p.
        # All output neurons need to be connected to something. Doesn't matter if not all inputs are connected.
        inp_ids = self.get_node_ids('input')
        for o_n in self.get_node_ids('output'):
            n_conns = np.random.randint(1, len(inp_ids))
            #chosen_inputs = np.random.choice(inp_ids, n_conns, replace=False)
            chosen_inputs = random.sample(inp_ids, n_conns)
            for i_n in chosen_inputs:
                if self.verbose: print('Connecting node %d to node %d' % (i_n, o_n))
                self.conn_genes.append({
                    'innov': self.innov,
                    'from': i_n,
                    'to': o_n,
                    'wgt': np.random.randn(), # From the normal distribution
                    'enabled': True
                })
                self.innov += 1
                
    def get_node_ids(self, layer='all'):
        if layer=='all':
            return [n['id'] for n in self.node_genes]
        else:
            return [n['id'] for n in self.node_genes if n['layer'] == layer]
    
    def get_conn_ids(self, only_enabled=False):
        if only_enabled:
            return [c['innov'] for c in self.conn_genes if c['enabled']]
        else:
            return [c['innov'] for c in self.conn_genes]
    
    def get_conn_gene(self, innov):
        found = [g for g in self.conn_genes if g['innov']==innov]
        if found:
            return found[0] # take it out of list
        else:
            raise ValueError('Innovation {} not found'.format(innov))
            
    def get_node_gene(self, idn):
        found = [g for g in self.node_genes if g['id']==idn]
        if found:
            return found[0]
        else:
            raise ValueError('Innovation {} not found'.format(idn))
    
    def get_connections(self, only_enabled=True):
        """
        Returns a set of tuples of connected nodes: (from, to)
        :return: a set of tuples of connected nodes
        """
        if only_enabled:
            return {(g['from'], g['to']) for g in self.conn_genes if g['enabled']}
        else:
            return {(g['from'], g['to']) for g in self.conn_genes}

    def add_connection(self):
        """
        Adds a connection gene creating a connection from the pool of unused pairwise node
        connections (not allowed self-connection or connection to the input layer.
        If it is not a recurrent network, then connections which would create a cycle are also not
        allowed.
        :return: None
        """
        existing_conns = self.get_connections(only_enabled=False)
        all_possible_conns = {(u['id'], v['id']) for u in self.node_genes for v in self.node_genes
                          if u['id'] != v['id'] # can't connect a node to itself
                          and v['layer'] != 'input' # can't connect TO an input node (cos input nodes are really just the data)
                          and u['layer'] != 'output'} # can't connect FROM an output layer (could relax this.)
        available_conns = all_possible_conns - existing_conns
        if self.verbose: print('Available connections: ' + str(available_conns))

        if not(self.recurrent):
            cycle_conns = set(filter(self.creates_cycle, available_conns))
            available_conns = available_conns - cycle_conns
            if self.verbose: print("Which wouldn't create a cycle: " + str(available_conns))

        if available_conns:
            chosen = random.sample(available_conns, 1)[0]
            if self.verbose: print('We chose: ' + str(chosen))
            self.conn_genes.append({
                'innov': self.innov,
                'from': chosen[0],
                'to': chosen[1],
                'wgt': np.random.randn(),  # From the normal distribution
                'enabled': True
            })
            self.innov += 1
        else:
            print('No new connections possible')

    def creates_cycle(self, test):
        """
        Returns true if the addition of the 'test' connection would create a cycle,
        assuming that no cycle already exists in the graph represented by 'connections'.
        Copied from: https://github.com/CodeReclaimers/neat-python/blob/master/neat/graphs.py
        """
        connections = self.get_connections(False)
        i, o = test
        if i == o:
            return True

        visited = {o}
        while True:
            num_added = 0
            for a, b in connections:
                if a in visited and b not in visited:
                    if b == i:
                        return True

                    visited.add(b)
                    num_added += 1

            if num_added == 0:
                return False

    def add_node(self):
        """
        Adds a new node by breaking an existing connection in two.
        The first of the two new connections has weight 1, and the second has the original weight.
        This means that when the node is first added it should have no effect.
        The original connection is kept but disabled.
        """
        chosen_gene = random.sample(self.conn_genes, 1)[0]
        # Create the new node
        new_node_id = len(self.node_genes)
        self.node_genes.append({
            'id': new_node_id,
            'layer': 'hidden',
            'activation': 'sigmoid',
            'bias': np.random.randn()
        })
        # Reorganise the connections
        new_conn1 = {
            'innov': self.innov,
            'from': chosen_gene['from'],
            'to': new_node_id,
            'wgt': 1, # first half of new split connection is 1.
            'enabled': True
        }
        self.innov += 1
        new_conn2 = {
            'innov': self.innov,
            'from': new_node_id,
            'to': chosen_gene['to'],
            'wgt': chosen_gene['wgt'],  # second half of new split connection is original weight
            'enabled': True
        }
        self.innov += 1
        self.conn_genes.append(new_conn1)
        self.conn_genes.append(new_conn2)
        # Disable old connection
        chosen_gene['enabled'] = False
        
    def split_node(self):
        """
        A new form of mutation invented by meee.
        The problem witk K.Stanley's two topographical mutations is they don't
        allow much sideways expansion. Eg. if you have a network with one input
        and one output and repeatedly add nodes they will all be in a single
        long chain...
        This split_node will mean A--B--C becomes:
          -B1-
        A<    >C
          -B2-
        How to keep effect the same initially?
        Could have act func
        """
        pass
    
    def mutate(self):
        """
        Could have a method which chooses one of the three mutation types
        (add_connection, add_node, alter_weight)
        based on probability of each. (Would this be an input or instance variables?)
        :return:
        """
        pass
    
    def randomise(self):
        """
        We shouldn't use this for real - just adding for testing.
        """
        for g in self.conn_genes:
            g['wgt'] = np.random.randn()
        for n in self.node_genes:
            n['bias'] = np.random.randn()
            
    def empty(self):
        """
        Returns an empty genome (ie. no node or connection genes)
        with same settings (recurrent, verbose) as this one.
        """
        return Genome(0, 0, recurrent=self.recurrent, verbose=self.verbose)

    def crossover(self, other: Genome):
        # Create a child genome with no connections
        child = Genome(self.n_in, self.n_out, init_conns=False, recurrent=self.recurrent, verbose=self.verbose)
        # Choose and add the connection genes
        self_innovs = self.get_conn_ids()
        other_innovs = other.get_conn_ids()
        all_innovs = set(self_innovs).union(set(other_innovs))
        for i in all_innovs:
            if i in self_innovs:
                if i in other_innovs:
                    # choose randomly
                    opt1 = self.get_conn_gene(i)
                    opt2 = other.get_conn_gene(i)
                    chosen = opt1.copy() if chooser(self, other) else opt2.copy()
                    if not(opt1['enabled'] and opt2['enabled']):
                        # Must be enabled in both parents to be enabled in offspring (for the moment)
                        chosen['enabled'] = False
                else:
                    chosen = self.get_conn_gene(i).copy()
            elif i in other_innovs:
                chosen = other.get_conn_gene(i).copy()
            child.conn_genes.append(chosen)
        # Now create the node genes based on what we need
        conns = child.get_connections(only_enabled=False) # list of (from,to) tuples
        need_nodes = np.unique(list(conns))
        have_nodes = child.get_node_ids()
        self_hidden_node_ids = self.get_node_ids(layer='hidden')
        other_hidden_node_ids = other.get_node_ids(layer='hidden')
        print("These are need nodes:")
        print(need_nodes)
        for n in need_nodes:
            print('Working on node {}'.format(n))
            if n in self_hidden_node_ids:
                if n in other_hidden_node_ids:
                    chosen = self.get_node_gene(n).copy() if chooser(self, other) else other.get_node_gene(n).copy()
                else:
                    chosen = self.get_node_gene(n).copy()
            elif n in other_hidden_node_ids:
                chosen = other.get_node_gene(n).copy()
            else:
                # If a needed node is not in either parent's hidden node list,
                # hopefully that is because it is an input/output node. If so
                # it should already have been initialised in the child.
                assert n in have_nodes, "Node {} isn't in child".format(n)
                continue
            child.node_genes.append(chosen.copy())
        return child
      
    # def zip_align(d1, d2, align_on):
    #     overlap = [(a,b) for a in d1 for b in d2 if a[align_on]=b[align_on]]
    #     d1_kw = [x[align_on] for x in d1]
    #     d2_kw = [x[align_on] for x in d2]
    #     d1_only = [(a,None) for a in d1 if a[align_on] not in d2_kw]
    #     d2_only = [(None,b) for b in d2 if b[align_on] not in d1_kw]

    def save(self, filename=None):
        if not filename:
            filename = 'genome_' + datetime.datetime.now().strftime("%d%b%Y_%I%p%M") + '.json'
        with open(filename, 'w') as savefile:
            json.dump([self.node_genes, self.conn_genes], savefile)
            
    @staticmethod
    def load(filename):
        with open(filename, 'r') as loadfile:
            data = json.load(loadfile)
        g = Genome(0,0,init_conns=False) # self.n_in and self.n_out will be wrong!
        g.node_genes = data[0]
        g.conn_genes = data[1]
        g.n_in = len(g.get_node_ids('input'))
        g.n_out = len(g.get_node_ids('output'))
        return g
    
    def __str__(self):
        out_str = ['%d: [%d]--%0.2f-->[%d]' % (g['innov'], g['from'], g['wgt'], g['to']) for g in self.conn_genes if g['enabled']]
        return '\n'.join(out_str)

def chooser(genome1: Genome, genome2: Genome):
    """
    A function to determine how genes are chosen when they are shared.
    ie. randomly or from fitter parent or some hybrid.
    Making a separate function just so it is easy to change it.
    Will initially choose randomly.
    Should return a bool since we are always choosing between two things.
    """
    return np.random.choice(2)==1


    
    
class NNFF(object):
    """
    This is initialised with a genome object and creates a working feedforward neural net.
    """
    genome = None

    def __init__(self, in_genome):
        assert not in_genome.recurrent, "NNET_FF can only cope with a non-recurrent genome."
        self.genome = in_genome
        self.layers = self.get_layers_ff()
        self.n_layers = len(self.layers)
        self.weights = self.get_weights()
        self.biases = self.get_biases()

    def get_biases(self):
        return [np.array([g['bias'] for g in self.genome.node_genes 
                          if g['id'] in layer]) 
                for layer in self.layers[1:]] # skip out input layer
        
    def get_weights(self):
        wgts_dict = {(gene['from'],gene['to']): gene['wgt'] for gene in self.genome.conn_genes if gene['enabled']}
        from_nodes = []
        layer_dicts = []
        for l in range(self.n_layers-1):
            from_nodes.extend(self.layers[l]) # this accumulates (because a connection can be from ANY previous layer)
            to_nodes = self.layers[l+1] # this does not accumulate
            wgts_mat = np.array([[wgts_dict.get((u, v), 0) for u in from_nodes] for v in to_nodes])
            layer_dicts.append({
                    'from_nodes': from_nodes.copy(), # create static copy
                    'to_nodes': to_nodes,
                    'wgts_mat': wgts_mat
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
        layers.append(seen)
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

            layers.append(tuple(sorted(next_layer))) # Sort nodes and convert to tuple
            seen = seen.union(next_layer)

        return layers
    
    def feedforward(self, a):
        """
        Because we might be reusing neurons from previous layers we can't do a 
        simple loop through the layer activations.
        Instead we'll keep a dict of node outputs so we can reuse them.
        """
        assert len(a)==len(self.layers[0]), "Input vector must be same size as input layer."
        node_vals = {node: val for (node,val) in zip(self.layers[0], a)} # Initialise the dict with inputs
        for wgt_dict, bias in zip(self.weights, self.biases):
            inp_vec = [node_vals[n] for n in wgt_dict['from_nodes']]
            weights = wgt_dict['wgts_mat']
            out_vec = sigmoid(np.dot(weights, inp_vec) + bias)
            node_vals.update( {node: val for (node,val) in zip(wgt_dict['to_nodes'],out_vec)} )
        
        return [node_vals[n] for n in self.genome.get_node_ids('output')]
        
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def netviz(genome):
    """
    Pass in a genome - prints a quick visualisation of the network.
    """
    genome.node_genes