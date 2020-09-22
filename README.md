# neat

Contents:
- genome.py: contains the Genome class which handles mutation etc.
- nnet.py: Contains the FeedForwardNet class which takes a Genome as input.
- population.py: TODO. Will contain Population class for handling pop-level stuff
- netviz.py: Contains functions for visualising a Genome.


8th July 2020:
- Changing genome so that rather than each node having
a bias, there will be a single bias input node (always 1)
which is connected to other nodes with a weight. This weight
mutates in order to change the bias.

To Consider:
- Aggregation and activation functions are presumably
chosen at time of node creation. Are these functions
then fixed or can they mutate? DECIDED - fixed as an inherent property of each node.
- Crossover:
-- shared genes: choose randomly or by fitness
-- disjoint/excess: take all, choose whether to take randomly or by fitness
-- reactivate diabled genes? if it is enabled in either parent?
- Meta-genetics. Probability of choosing each activation func (or each
type of mutation) could be a property of the genome. This meta 
property could also evolve (with what mechanics??)
-- motivation: genomes whic use the mod() activation function
a lot will produce a certain *style* of images even though the 
content of the image differs. Style is like a meta property that 
could be shared within certain 'bloodlines'.
-- so could average actfunc probabilities at every crossover?


TO DO!:
- PROBLEM: currently if the same innovation (eg connection from node 1 to node 7)
appears separately in different individuals because of different mutations,
it will have different innovation numbers. This means that when crossover
takes place there can be duplication because the same connection has two
different innovation numbers.
- SOLUTION: annoying but I think we will have to have a global dict of
connections with the keys as (from_node,to_node) tuples and the values
as innovation numbers. Then when a mutation wants to add a connection,
rather than just incrementing the innovation number and labelling the new conn
with that, it should first check the dictionary to see if that innovation
has already occured and if so, use the existing innovation number to label it.
Node numbers will also need to be global!

better to reorganise methods as:
make_connection(from, to, wgt): handles dictionary lookup and creating the gene
add_random_connection(): selects an allowed connection and calls make_connection to create it.
init_connections(): use existing sampling routine then call make_connection()

Novelty search
- Stanley's idea is just to keep a record of every image we have seen and 
assign fitness based on how different an image is from anything we have seen
before.
- But how do we assess difference? Most obvious is pixel-by-pixel value difference
(cartesian distance between vectors of pixel values).
- But an image of vertical b&w stipes shifted one pixel to the right would be 100%
different.
- More useful to capture something about the structure of the image.
- So could put it through a CNN and measure difference of output vectors.
- But CNN not likely to be good without training (so could train on real images orr..?)

Notes on pretrained tensorflow assesor:
- Using TF network pretrained on ImageNet (1001 classes of image)
- Plan is simply to use max probability as our fitness rating (ie. 
we don't really care what an image looks like as long as it looks 
like something.)
- But in fact this quickly converges on getting very hgh probabilities
for classes like "jellyfish" or "velvet". And evolution then pretty
much stops (any mutations introducing a new thing are quickly quashed)
- So this is where we need speciation, and the class labels provide an
easy means to achieve it.
- For starters I plan just to boost the rating of any image where the
"winning" class is one we haven't seen before.
- This works ok for rewarding rarity but we still tend to get stuck with 
a lack of diversity. So speciation is going to become necessary. The problem
is that whenever a new class is introduced, it gets a very high rating and the
population rapidly moves completely towards that class, then the ratings fall
quickly until  new class is introduced. So we end up just flippling between classes

Possibility of using max as agg_func:
https://stackoverflow.com/questions/41164305/numpy-dot-product-with-max-instead-of-sum
basically it will be much slower than np.dot.