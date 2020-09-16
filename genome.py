from __future__ import annotations # enables type hints of self object
import random
import datetime
import json
import yaml

import numpy as np

from funcs import get_funcs, create_args

INNOV = 0
CONN_DICT = dict()
#NODE_DICT = dict()

with open('config.yaml','r') as f:
    CONFIG = yaml.safe_load(f)

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
        #self.fitness = 0 
        self.metadata = {'fitness': 0} # For any extra info we may want to attach to the genome. Init fitness to zero.
        #self.allowed_act_funcs = ['sigmoid', 'relu', 'tanh', 'sin', 'abs']
        #self.allowed_act_funcs = ['round','mod']
        self.allowed_act_funcs = get_funcs('names')
        #self.allowed_act_funcs = ['tanh']
        # default could be everything returned by: get_funcs('names')
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
            #act_func = random.choice(['gaussian','sigmoid'])
            act_func = 'sigmoid'
            act_args = create_args(get_funcs(act_func))
            self.node_genes.append({
                'id': i,
                'layer': 'output',
                'agg_func': 'sum',
                'act_func': act_func,
                'act_args': act_args
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
                self.make_connection((i_n, o_n))
                
                
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
            self.make_connection(chosen)
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
        act_func = random.choice(self.allowed_act_funcs)
        # NB Otoro uses tanh for all but the output layer.
        act_args = create_args(get_funcs(act_func)) # gets some random values for any extra args required
        self.node_genes.append({
            'id': new_node_id,
            'layer': 'hidden',
            'agg_func': 'sum',
            'act_func': act_func,
            'act_args': act_args
        })
        # Reorganise the connections
        self.make_connection((chosen_gene['from'], new_node_id), wgt=1)
        self.make_connection((new_node_id, chosen_gene['to']), wgt=chosen_gene['wgt'])
        # Disable the original connection
        chosen_gene['enabled'] = False
        
    def make_connection(self, path, wgt=None):
        """
        Appends a connection to self.conn_genes using the 
        appropriate innovation number (incremented if
        connection not seen before, otherwise using the
        innovation number of when it was first made).
        """
        global INNOV
        global CONN_DICT
        if wgt==None: wgt = np.random.normal() # From the normal distribution
        if path in CONN_DICT:
                innov = CONN_DICT[path]
        else:
            innov = INNOV
            CONN_DICT[path] = innov
            INNOV += 1
        # add the connection gene.
        self.conn_genes.append({
            'innov': innov,
            'from': path[0],
            'to': path[1],
            'wgt': wgt,  
            'enabled': True
        })
        
    def disable_random_conn(self):
        """
        A mutation type which disables a random connection gene.
        Generally have quite a low probability of this type of
        mutation. But it is useful because it offers a way to
        reduce complexity. Would we be better off removing the gene
        entirely rather than disabling?
        """
        # We need to not end up with any "stranded" nodes
        # ie. nodes not in the input layer which don't have iny inputs.
        # I thiiink any conn where the 'to' node has other inputs is safe to delete.
        # That way we shouldn't get any stranded nodes.
        # So we might need to go through all the connections to find one that is ok to delete.
        chosen_genes = random.sample(self.conn_genes, len(self.conn_genes))
        for cg in chosen_genes:
            if np.sum( np.array([t[1] for t in self.get_connections()]) == cg['to']) >=2:
                cg['enabled'] = False
                break
        if self.verbose: print('No disable-able connection found')
        
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
    
    def flip_weight(self, n_to_flip=1):
        chosen = random.sample(self.conn_genes, n_to_flip)
        for conn in chosen:
            # flip the sign
            conn['wgt'] = -conn['wgt']
        
    
    def mutate(self):
        """
        Amethod which chooses one of the three mutation types
        (add_connection, add_node, alter_weight)
        based on probability of each. (Would this be an input or instance variables?)
        For now using fixed probabilities:
            add_connection should have higher probability than add_node
            (because we need more connections than nodes)
            alter_weight should have the highest probability.
        Would be nice to have these alter with age. Young one want to
        add nodes/connections fast, older ones whould focus on altering weights.
        """

        func_dict = {'alter_wgt': self.alter_weight,
                     'flip_wgt': self.flip_weight,
                     'add_connection': self.add_connection,
                     'add_node': self.add_node,
                     'disable_connection': self.disable_random_conn}
        mutation_types = CONFIG['mutation_types']
        options = [func_dict[m['func']] for m in mutation_types]
        probs = [m['prob'] for m in mutation_types]
        chosen_mutation = np.random.choice(options, p=probs)
        chosen_mutation() # Execute the chosen mutation.

        
    
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
                    # Innovation is in both parents so choose randomly
                    opt1 = self.get_conn_gene(i)
                    opt2 = other.get_conn_gene(i)
                    if child.recurrent:
                        options = [opt1, opt2]
                    else:
                        # We need to check if either option would create a cycle in the child
                        opt1_ok = not( child.creates_cycle((opt1['from'], opt1['to'])) )
                        opt2_ok = not( child.creates_cycle((opt2['from'], opt2['to'])) )
                        options = []
                        if opt1_ok: options.append(opt1)
                        if opt2_ok: options.append(opt2)
                    if options:
                        chosen = random.choice(options).copy()
                    else:
                        print('Neither patent gene inherited as either would create a cycle')
                        continue
                    if not(opt1['enabled']) or not(opt2['enabled']):
                        # Stanley: "there’s a preset chance that an inherited gene is disabled if it is disabled in either parent."
                        if random.random()<0.6: # hard coded 60% chance. not ideal.
                            chosen['enabled'] = False
                else:
                    # Innovation is only in the "self" parent
                    chosen = self.get_conn_gene(i).copy()
            elif i in other_innovs:
                # Innovation is only in the "other" parent
                chosen = other.get_conn_gene(i).copy()
            # Check the proposed doesn't create a cycle in the child
            chosen_ok = not( child.creates_cycle((chosen['from'], chosen['to'])) ) if not(child.recurrent) else True
            if chosen_ok:
                child.conn_genes.append(chosen)
            else:
                print('Gene not inherited as would create cycle')
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
        while random.random()<mut_rate:
            # TODO this was an if. HAve made a while. THink about this.
            child.mutate()
        return child
    
    def get_fitness(self, raw=False):
        if raw:
            # Get raw_fitness or if it doesn't exist return fitness
            return self.metadata.get('raw_fitness', self.metadata.get('fitness'))
        else:
            return self.metadata.get('fitness')

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
