from __future__ import annotations # enables type hints
import random
import datetime
import json

import numpy as np

INNOV = 0
CONN_DICT = dict()

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
        self.fitness = 0 # init fitness to zero.
        # Add the input and output nodes to the list of node_genes
        self.node_genes = []
        for i in range(n_in):
            self.node_genes.append({
                'id': i,
                'layer': 'input',
                'agg_func': None,
                'act_func': None,
            })
        for i in range(n_in, n_in+n_out):
            self.node_genes.append({
                'id': i,
                'layer': 'output',
                'agg_func': 'sum',
                'act_func': 'sigmoid' #output function #'sigmoid'
            })
        self.conn_genes = []
        if init_conns:
            self.init_conns()
        
    def init_conns(self):
        # Now make some connections between inputs and outputs according to the probabilities in p.
        # All output neurons need to be connected to something. Doesn't matter if not all inputs are connected.
        global INNOV
        global CONN_DICT
        inp_ids = self.get_node_ids('input')
        for o_n in self.get_node_ids('output'):
            n_conns = np.random.randint(1, len(inp_ids))
            #chosen_inputs = np.random.choice(inp_ids, n_conns, replace=False)
            chosen_inputs = random.sample(inp_ids, n_conns)
            for i_n in chosen_inputs:
                if self.verbose: print('Connecting node %d to node %d' % (i_n, o_n))
                # check if it already exists in the connection dictionary
                if (i_n, o_n) in CONN_DICT:
                    innov = CONN_DICT[(i_n, o_n)]
                else:
                    innov = INNOV
                    CONN_DICT[(i_n, o_n)] = innov
                    INNOV += 1
                # add the gene
                self.conn_genes.append({
                    'innov': INNOV,
                    'from': i_n,
                    'to': o_n,
                    'wgt': np.random.randn(), # From the normal distribution
                    'enabled': True
                })
                
                
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
        global INNOV
        global CONN_DICT
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
            # check if it already exists in the connection dictionary
            if chosen in CONN_DICT:
                innov = CONN_DICT[chosen]
            else:
                innov = INNOV
                CONN_DICT[chosen] = innov
                INNOV += 1
            # add the connection gene.
            self.conn_genes.append({
                'innov': INNOV,
                'from': chosen[0],
                'to': chosen[1],
                'wgt': np.random.randn(),  # From the normal distribution
                'enabled': True
            })
            INNOV += 1
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
        global INNOV
        chosen_gene = random.sample(self.conn_genes, 1)[0]
        # Create the new node
        new_node_id = len(self.node_genes)
        self.node_genes.append({
            'id': new_node_id,
            'layer': 'hidden',
            'agg_func': 'sum',
            'act_func': random.sample(['sin','tanh'],1)[0], # going to use tanh for all but output layer at the mo (following otoro)
        })
        # Reorganise the connections
        new_conn1 = {
            'innov': INNOV,
            'from': chosen_gene['from'],
            'to': new_node_id,
            'wgt': 1, # first half of new split connection is 1.
            'enabled': True
        }
        INNOV += 1
        new_conn2 = {
            'innov': INNOV,
            'from': new_node_id,
            'to': chosen_gene['to'],
            'wgt': chosen_gene['wgt'],  # second half of new split connection is original weight
            'enabled': True
        }
        INNOV += 1
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
    
    def alter_weight(self, n_to_alter=1):
        chosen = random.sample(self.conn_genes, n_to_alter)
        for conn in chosen:
            # peturb by a number selected from the normal distribution.
            conn['wgt'] += float(np.random.randn(1))
    
    def mutate(self):
        """
        Could have a method which chooses one of the three mutation types
        (add_connection, add_node, alter_weight)
        based on probability of each. (Would this be an input or instance variables?)
        For now using fixed probabilities:
            add_connection should have higher probability than add_node
            (because we need more connections than nodes)
            alter_weight should have the highest probability.
        """
        wgt_prob = 0.5
        conn_prob = 0.3
        node_prob = 0.2
        r = random.random()
        if r <= wgt_prob:
            self.alter_weight
        elif r <= wgt_prob + conn_prob:
            self.add_connection()
        elif r <= wgt_prob + conn_prob + node_prob:
            self.add_node()
    
    def randomise_weights(self):
        """
        We shouldn't use this for real - just adding for testing.
        """
        for g in self.conn_genes:
            g['wgt'] = np.random.randn()
            
    def empty(self):
        """
        Returns an empty genome (ie. no node or connection genes)
        with same settings (recurrent, verbose) as this one.
        """
        return Genome(0, 0, recurrent=self.recurrent, verbose=self.verbose)

    def crossover(self, other: Genome, mut_rate=0):
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
                    if not(opt1['enabled']) or not(opt2['enabled']):
                        # Stanley: "there’s a preset chance that an inherited gene is disabled if it is disabled in either parent."
                        if random.random()<0.5: # hard coded 50% chance. not ideal.
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
        for n in need_nodes:
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
        if random.random()<mut_rate:
            child.mutate()
        return child
      
    # def zip_align(d1, d2, align_on):
    #     overlap = [(a,b) for a in d1 for b in d2 if a[align_on]=b[align_on]]
    #     d1_kw = [x[align_on] for x in d1]
    #     d2_kw = [x[align_on] for x in d2]
    #     d1_only = [(a,None) for a in d1 if a[align_on] not in d2_kw]
    #     d2_only = [(None,b) for b in d2 if b[align_on] not in d1_kw]

    def save(self, filename=None):
        if not filename:
            filename = "./output/genome_" + datetime.datetime.now().strftime("%d%b%Y_%I%p%M") + '.json'
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
        # out_str = ['%d: [%d]--%0.2f-->[%d]' % (g['innov'], g['from'], g['wgt'], g['to']) 
        #            for g in self.conn_genes if g['enabled']
        #            else '%d: [%d]--%0.2f-->[%d] (DISABLED)' % (g['innov'], g['from'], g['wgt'], g['to'])]
        out_str = ["{:d}: [{:d}]--{:0.2f}-->[{:d}] {active}"
                   .format(g['innov'], g['from'], g['wgt'], g['to'], active='' if g['enabled'] else '(DISABLBED)')
                   for g in self.conn_genes]
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
