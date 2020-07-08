# neat

8th July 2020:
- Changing genome so that rather than each node having
a bias, there will be a single bias input node (always 1)
which is connected to other nodes with a weight. This weight
mutates in order to change the bias.

To Consider:
- Aggregation and activation functions are presumably
chosen at time of node creation. Are these functions
then fixed or can they mutate? 
- Crossover:
-- shared genes: choose randomly or by fitness
-- disjoint/excess: take all, choose whether to take randomly or by fitness
-- reactivate diabled genes? if it is enabled in either parent?