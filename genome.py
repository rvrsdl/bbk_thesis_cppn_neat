from __future__ import annotations  # enables type hints of self object
import random
import datetime
import json

import numpy as np

import funcs

INNOV = 0
CONN_DICT = dict()


class Genome(object):
    """
    This object manages the genome and handles mutation.
    """

    def __init__(self, n_in: int, n_out: int, recurrent: bool = False,
                 verbose: bool = False, init_conns: bool = True, **kwargs):
        """
        Initialise the genes for the basic network in which each output node
        is connected to a random selection of between one and all the input
        nodes.
        :param n_in: Number of input nodes (including bias)
        :param n_out: Number of output nodes.
        :param recurrent: Whether the network is allowed to be recurrent.
        :param verbose: Print status messages (mainly for debugging)
        """
        # Set public attributes
        self.n_in = n_in
        self.n_out = n_out
        self.recurrent = recurrent
        self.verbose = verbose
        self.metadata = {'fitness': 0}  # For any extra info we may want to attach to the genome. Init fitness to zero.

        # Set private attributes
        defaults = self._default_settings()
        self._activation_funcs = kwargs.get('activation_funcs', defaults.get('activation_funcs'))
        self._output_funcs = kwargs.get('output_funcs', defaults.get('output_funcs'))
        self._mutation_types = kwargs.get('mutation_types', defaults.get('mutation_types'))
        self._disable_prob = kwargs.get('disable_prob', defaults.get('disable_prob'))

        # Create a list of input and output node genes
        self._node_genes = []
        for i in range(n_in):
            self._create_node_gene(i, 'input')
        for i in range(n_in, n_in + n_out):
            self._create_node_gene(i, 'output')

        # Create a list of connection genes
        self._conn_genes = []
        if init_conns:  # TODO - this seems weird.
            self.init_conns()

    @staticmethod
    def _default_settings() -> dict:
        """
        Returns a dict of defualt settings for use in __init__
        in case explicit settings aren't provided.
        """
        settings = {
            'activation_funcs': funcs.get_funcs('names'),
            'output_funcs': ['sigmoid', 'gaussian'],
            'mutation_types': [
                {'func': 'alter_wgt', 'prob': 0.4},
                {'func': 'flip_wgt', 'prob': 0.1},
                {'func': 'add_connection', 'prob': 0.3},
                {'func': 'add_node', 'prob': 0.1},
                {'func': 'disable_connection', 'prob': 0.1}
            ],
            'disable_prob': 0.6
        }
        return settings

    def init_conns(self) -> None:
        """
        Initialises connections between the input and output nodes.
        (At this point there should be no hidden layer nodes as this
        should be called during initialisation.)
        """
        # All output neurons need to be connected to something. 
        # Doesn't matter if not all inputs are connected.
        inp_ids = self.get_node_ids('input')
        for o_n in self.get_node_ids('output'):
            n_conns = random.randint(1, len(inp_ids))
            chosen_inputs = random.sample(inp_ids, n_conns)
            for i_n in chosen_inputs:
                if self.verbose:
                    print('Connecting node %d to node %d' % (i_n, o_n))
                self._create_conn_gene((i_n, o_n))

    def get_node_ids(self, layer: str = 'all') -> list:
        """
        Returns the node ID numbers in a particular layer (input, hidden or 
        output)
        """
        if layer == 'all':
            return [n['id'] for n in self._node_genes]
        else:
            return [n['id'] for n in self._node_genes if n['layer'] == layer]

    def get_conn_ids(self, only_enabled: bool = False) -> list:
        """
        Returns the connection ID numbers (aka the innovation numbers).
        It can optionally do it only for the enabled ones.
        """
        if only_enabled:
            return [c['innov'] for c in self._conn_genes if c['enabled']]
        else:
            return [c['innov'] for c in self._conn_genes]

    def get_conn_gene(self, innov: int) -> dict:
        """
        Returns the information dict of a particular connection gene.
        """
        found = [g for g in self._conn_genes if g['innov'] == innov]
        if found:
            return found[0]  # take it out of list
        else:
            raise ValueError('Innovation {} not found'.format(innov))

    def get_node_gene(self, idnum: int) -> dict:
        """
        Returns the information dict of a particula node gene.
        """
        found = [g for g in self._node_genes if g['id'] == idnum]
        if found:
            return found[0]
        else:
            raise ValueError('Innovation {} not found'.format(idnum))

    def get_connections(self, only_enabled: bool = True) -> set:
        """
        Returns a set of tuples of connected nodes: (from, to)
        By default it only returns the enabled connections
        """
        if only_enabled:
            return {(g['from'], g['to']) for g in self._conn_genes if g['enabled']}
        else:
            return {(g['from'], g['to']) for g in self._conn_genes}

    def get_wgts_dict(self) -> dict:
        """
        Returns a dict of weights with keys as tuples of the (from, to) nodes.
        """
        return {(gene['from'], gene['to']): gene['wgt'] for gene in self._conn_genes if gene['enabled']}

    def add_node(self) -> None:
        """
        Adds a new node by breaking an existing connection in two.
        The first of the two new connections has weight 1, and the second has 
        the original weight.
        This means that when the node is first added it should have no effect.
        The original connection is kept but disabled.
        """
        chosen_gene = random.sample(self._conn_genes, 1)[0]
        # Create the new node
        new_node_id = len(self._node_genes)
        self._create_node_gene(new_node_id, 'hidden')
        # Reorganise the connections
        self._create_conn_gene((chosen_gene['from'], new_node_id), wgt=1)
        self._create_conn_gene((new_node_id, chosen_gene['to']), wgt=chosen_gene['wgt'])
        # Disable the original connection
        chosen_gene['enabled'] = False

    def add_connection(self) -> None:
        """
        Adds a connection gene, creating a connection from the pool of unused 
        pairwise node connections.
        We are not allowed self-connections or connections to the input layer.
        If it is not a recurrent network, then connections which would create 
        a cycle are also not allowed.
        """
        existing_conns = self.get_connections(only_enabled=False)
        all_possible_conns = {(u['id'], v['id']) for u in self._node_genes for v in self._node_genes
                              if u['id'] != v['id']  # can't connect a node to itself
                              and v['layer'] != 'input'  # can't connect TO an input node
                              and u['layer'] != 'output'}  # can't connect FROM an output layer
        available_conns = all_possible_conns - existing_conns
        if self.verbose: print('Available connections: ' + str(available_conns))

        if not self.recurrent:
            cycle_conns = set(filter(self._check_for_cycle, available_conns))
            available_conns = available_conns - cycle_conns
            if self.verbose: print("Which wouldn't create a cycle: " + str(available_conns))

        if available_conns:
            chosen = random.sample(available_conns, 1)[0]
            if self.verbose: print('We chose: ' + str(chosen))
            self._create_conn_gene(chosen)  # Actually add a new connection gene
        else:
            print('No new connections possible')

    def _check_for_cycle(self, candidate: tuple):
        """
        Returns true if adding the candidate connection would create a cycle
        in the graph of connections.
        """
        connections = self.get_connections(False)
        _from, _to = candidate
        if _from == _to:
            return True

        seen = {_to}
        while True:
            num_added = 0
            for a, b in connections:
                if a in seen and b not in seen:
                    if b == _from:
                        return True
                    seen.add(b)
                    num_added += 1

            if num_added == 0:
                return False

    def _create_node_gene(self, idnum: int, layer: str) -> None:
        """
        Appends a node gene to self._node_genes. Chooses an activation function
        at random from the available functions for the hidden or output layers.
        """
        if layer == 'input':
            act_func_name = None
            act_func_args = None
        else:
            if layer == 'output':
                act_func_name = random.choice(self._output_funcs)
            else:
                act_func_name = random.choice(self._activation_funcs)
            act_func_args = funcs.create_args(funcs.get_funcs(act_func_name))
        self._node_genes.append({
            'id': idnum,
            'layer': layer,
            'agg_func': 'sum',
            'act_func': act_func_name,
            'act_args': act_func_args
        })

    def _create_conn_gene(self, path: tuple, wgt: object = None) -> None:
        """
        Appends a connection to self._conn_genes using the 
        appropriate innovation number (incremented if
        connection not seen before, otherwise using the
        innovation number of when it was first made).
        """
        global INNOV
        global CONN_DICT
        if wgt is None:
            wgt = np.random.uniform(-5, 5)
            # TODO: was using np.random.normal() # From the normal distribution
        if path in CONN_DICT:
            innov = CONN_DICT[path]
        else:
            innov = INNOV
            CONN_DICT[path] = innov
            INNOV += 1
        # add the connection gene.
        self._conn_genes.append({
            'innov': innov,
            'from': path[0],
            'to': path[1],
            'wgt': wgt,
            'enabled': True
        })

    def _disable_random_conn(self) -> None:
        """
        A mutation which disables a random connection gene.
        We will generally have quite a low probability of this type of 
        mutation. But it is useful because it offers a way to reduce complexity.
        """
        # We need to not end up with any "stranded" nodes
        # ie. nodes not in the input layer which don't have iny inputs.
        # Any connection where the 'to' node has other inputs is safe to delete.
        # That way we shouldn't get any stranded nodes.
        # We might need to go through all the connections to find one that is 
        # ok to delete.
        chosen_genes = random.sample(self._conn_genes,
                                     len(self._conn_genes))  # get a shuffled list of the connection genes
        to_connections = np.array([t[1] for t in self.get_connections()])
        for cg in chosen_genes:
            if np.sum(to_connections == cg['to']) >= 2:
                cg['enabled'] = False
                break
        if self.verbose: print('No disable-able connection found')

    def _alter_weight(self, n_to_alter: int = 1) -> None:
        """
        A mutation which alters the weight of one or maore random connection
        genes. It peturbs the weight by an amount selected from the normal
        distribution. This will usually be the most common type of mutation.
        """
        chosen = random.sample(self._conn_genes, n_to_alter)
        for conn in chosen:
            # peturb by a number selected from the normal distribution.
            conn['wgt'] += float(np.random.normal())

    def _flip_weight(self, n_to_flip: int = 1) -> None:
        """
        A mutation which inverts (positive/negative) the weight of one or more
        random connection genes. Should be a fairly rare mutation.
        """
        chosen = random.sample(self._conn_genes, n_to_flip)
        for conn in chosen:
            # flip the sign
            conn['wgt'] = -conn['wgt']

    def mutate(self):
        """
        Applies one of the mutation types based on the probabilities set
        during initialisation.
        """
        # Make a func dict mapping from the mutation type names to the actual functions
        func_dict = {'alter_wgt': self._alter_weight,
                     'flip_wgt': self._flip_weight,
                     'add_connection': self.add_connection,
                     'add_node': self.add_node,
                     'disable_connection': self._disable_random_conn}
        options = [func_dict[m['func']] for m in self._mutation_types]
        probs = [m['prob'] for m in self._mutation_types]
        chosen_mutation = np.random.choice(options, p=probs)
        chosen_mutation()  # Execute the chosen mutation.

    def crossover(self, other: Genome, mut_rate: float = 0) -> Genome:
        """
        Creates an offspring genome from this and another parent.
        Some of the offsprings genes are mutated (controlled by the mut_rate
        parameter).
        The mut_rate should be between 0 and 1. There is an exponentially
        decreasing chance of multiple mutations:  if mut_rate is 0.5 then the 
        chance of one mutation is 0.5, of two is 0.25, of three is 0.125 etc.
        """
        child = self._empty()  # Create a child genome with no connections and the same settings as this one
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
                        opt1_ok = not (child._check_for_cycle((opt1['from'], opt1['to'])))
                        opt2_ok = not (child._check_for_cycle((opt2['from'], opt2['to'])))
                        options = []
                        if opt1_ok: options.append(opt1)
                        if opt2_ok: options.append(opt2)
                    if options:
                        chosen = random.choice(options).copy()
                    else:
                        print('Neither parent gene inherited as either would create a cycle')
                        continue
                    if not (opt1['enabled']) or not (opt2['enabled']):
                        # Stanley: "there’s a preset chance that an inherited gene is disabled
                        # if it is disabled in either parent."
                        if random.random() < self._disable_prob:  # hard coded 60% chance. not ideal.
                            chosen['enabled'] = False
                else:
                    # Innovation is only in the "self" parent
                    chosen = self.get_conn_gene(i).copy()
            elif i in other_innovs:
                # Innovation is only in the "other" parent
                chosen = other.get_conn_gene(i).copy()
            # Check the proposed doesn't create a cycle in the child
            chosen_ok = not (child._check_for_cycle((chosen['from'], chosen['to']))) if not child.recurrent else True
            if chosen_ok:
                child._conn_genes.append(chosen)
            else:
                print('Gene not inherited as would create cycle')
        # Now create the node genes based on what we need
        conns = child.get_connections(only_enabled=False)  # list of (from,to) tuples
        need_nodes = np.unique(list(conns))
        have_nodes = child.get_node_ids()
        self_hidden_node_ids = self.get_node_ids(layer='hidden')
        other_hidden_node_ids = other.get_node_ids(layer='hidden')
        for n in need_nodes:
            if n in self_hidden_node_ids:
                if n in other_hidden_node_ids:
                    chosen = random.choice([self.get_node_gene(n).copy(), other.get_node_gene(n).copy()])
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
            child._node_genes.append(chosen.copy())
        while random.random() < mut_rate:
            # NB this is a WHILE not an IF. So if mut_rate is 0.5 then the 
            # chance of one mutation is 0.5, of 2 is 0.25, of 3 is 0.125 etc.
            child.mutate()
        return child

    def get_fitness(self, raw: bool = False) -> float:
        """
        Returns the fitness of the genome. Optionally returns the raw_fitness
        if it exists (eg. for use with evaluators which penalise lack of novelty:
        the image is assigned a raw fitness based on its merit, but the final
        fitness is lower if it isn't novel.
        """
        if raw:
            # Get raw_fitness or if it doesn't exist return fitness
            return self.metadata.get('raw_fitness', self.metadata.get('fitness'))
        else:
            return self.metadata.get('fitness')

    def randomise_weights(self) -> None:
        """
        Shouldn't be used during an evolutionary run, but it is useful for
        testing.
        """
        for g in self._conn_genes:
            g['wgt'] = np.random.normal()

    def _empty(self) -> Genome:
        """
        Returns a genome with the same settings as this one but does not
        initiate any connections. Used by crossover method.
        """
        return Genome(self.n_in, self.n_out, init_conns=False,
                      recurrent=self.recurrent, verbose=self.verbose,
                      activation_funcs=self._activation_funcs,
                      output_funcs=self._output_funcs,
                      mutation_types=self._mutation_types,
                      disable_prob=self._disable_prob)

    def save(self, filename: str = None) -> None:
        # TODO: use save loc from config and match image saving protocol.
        if not filename:
            filename = "./output/genome_" + datetime.datetime.now().strftime("%d%b%Y_%I%p%M") + '.json'
        with open(filename, 'w') as savefile:
            json.dump([self._node_genes, self._conn_genes], savefile)

    @staticmethod
    def load(filename: str) -> Genome:
        with open(filename, 'r') as loadfile:
            data = json.load(loadfile)
        g = Genome(0, 0, init_conns=False)
        g._node_genes = data[0]
        g._conn_genes = data[1]
        g.n_in = len(g.get_node_ids('input'))
        g.n_out = len(g.get_node_ids('output'))
        return g

    def __str__(self):
        # out_str = ['%d: [%d]--%0.2f-->[%d]' % (g['innov'], g['from'], g['wgt'], g['to']) 
        #            for g in self.conn_genes if g['enabled']
        #            else '%d: [%d]--%0.2f-->[%d] (DISABLED)' % (g['innov'], g['from'], g['wgt'], g['to'])]
        out_str = ["{:d}: [{:d}]--{:0.2f}-->[{:d}] {active}"
                       .format(g['innov'], g['from'], g['wgt'], g['to'], active='' if g['enabled'] else '(DISABLBED)')
                   for g in self._conn_genes]
        return '\n'.join(out_str)
