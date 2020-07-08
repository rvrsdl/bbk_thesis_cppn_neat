"""
Script to make and save (json) the parent genomes in Fig.4 of
original NEAT paper.
NB all innovation and node numbers are one less than in Fig.4
as we are uzing zero-ased indexing.
"""
import nnet
import netviz as v

parent1 = nnet.Genome(3,1,init_conns=False)
parent1.node_genes.append({
    'id': 4,
    'layer': 'hidden',
    'agg_func': None,
    'act_func': None,
    })
parent1.conn_genes.append({
    'innov': 0,
    'from': 0,
    'to': 3,
    'wgt': 0.5,
    'enabled': True
    })
parent1.conn_genes.append({
    'innov': 1,
    'from': 1,
    'to': 3,
    'wgt': 0.5,
    'enabled': False
    })
parent1.conn_genes.append({
    'innov': 2,
    'from': 2,
    'to': 3,
    'wgt': 0.5,
    'enabled': True
    })
parent1.conn_genes.append({
    'innov': 3,
    'from': 1,
    'to': 4,
    'wgt': 0.5,
    'enabled': True
    })
parent1.conn_genes.append({
    'innov': 4,
    'from': 4,
    'to': 3,
    'wgt': 0.5,
    'enabled': True
    })
parent1.conn_genes.append({
    'innov': 7,
    'from': 0,
    'to': 4,
    'wgt': 0.5,
    'enabled': True
    })

parent2 = nnet.Genome(3,1,init_conns=False)
parent2.node_genes.append({
    'id': 4,
    'layer': 'hidden'
    })
parent2.node_genes.append({
    'id': 5,
    'layer': 'hidden'
    })
parent2.conn_genes.append({
    'innov': 0,
    'from': 0,
    'to': 3,
    'wgt': 0.5,
    'enabled': True
    })
parent2.conn_genes.append({
    'innov': 1,
    'from': 1,
    'to': 3,
    'wgt': 0.5,
    'enabled': False
    })
parent2.conn_genes.append({
    'innov': 2,
    'from': 2,
    'to': 3,
    'wgt': 0.5,
    'enabled': True
    })
parent2.conn_genes.append({
    'innov': 3,
    'from': 1,
    'to': 4,
    'wgt': 0.5,
    'enabled': True
    })
parent2.conn_genes.append({
    'innov': 4,
    'from': 4,
    'to': 3,
    'wgt': 0.5,
    'enabled': False
    })
parent2.conn_genes.append({
    'innov': 5,
    'from': 4,
    'to': 5,
    'wgt': 0.5,
    'enabled': True
    })
parent2.conn_genes.append({
    'innov': 6,
    'from': 5,
    'to': 3,
    'wgt': 0.5,
    'enabled': True
    })
parent2.conn_genes.append({
    'innov': 8,
    'from': 2,
    'to': 4,
    'wgt': 0.5,
    'enabled': True
    })
parent2.conn_genes.append({
    'innov': 9,
    'from': 0,
    'to': 5,
    'wgt': 0.5,
    'enabled': True
    })

parent1.save(filename='stanley_parent1.json')
parent2.save(filename='stanley_parent2.json')
