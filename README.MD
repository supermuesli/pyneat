# pyneat

A python module that implements neural networks capable of applying NEAT.

#### python 3.7.1+ dependencies
- numpy
- graphviz
- matplotlib
- ujson

#### software dependencies
- graphviz

#### what to expect of pyneat
- easily generate arbitrary neural networks made for reinforcement learning
- easily save and load models using json dumps
- 

#### usage
```python
# this demo implements a neural network that learns a dataset by applying NEAT.

import pyneat as pn

# this demo implements a neural network that learns a dataset by applying NEAT.

# define a neuralnet with 5 input_nodes and 1 output_node.
# set mutationrate to 1% (defaults to 5%),
# setting short_term_memory=False disables recurrent flow in the network.
# set the name to 'docs/xor' (defaults to current date-timestamp).
nn = pn.Neuralnet(5, 1, mutation_rate=0.01, short_term_memory=False, name='docs/xor')

# load the model (we already have a json dump. after each good mutation,
# the model will be dumped into a json). this step is not neccesary.
#nn.load(nn.name)

# a dataset. this one yields XOR.
inputs = [[1,1,0,0,0], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1], [1,0,0,0,1], [0,0,0,1,1], [0,1,0,0,0], [1,1,1,1,1], [1,0,0,1,0]]
outputs = [[0], [1], [1], [1], [0], [0], [1], [0], [0]]

# define a fitness function to train with. any positively growing 
# function with respect to positively counting attributes will suffice.
def fitness(): return 1000 - pn.MSE(nn, inputs, outputs)

# mount the fitness function onto the model.
nn.fitness_func = fitness

# train the model, plot the fitness and show the topology.
nn.train(100, plot=True, show=True)

# use the model.
print('0 xor 0 xor 0 xor 1 xor 0 <=> ', nn.forward([0,0,0,1,0]))
```