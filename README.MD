# pyneat

A python module that implements neural networks capable of applying NEAT.

#### python3 requirements
- numpy

#### what to expect of pyneat
- easily generate arbitrary neural networks made for reinforcement learning
- easily save and load models using json

#### usage
```python
# this demo implements a neural network that learns a dataset by applying NEAT.

import pyneat as pn

# a neuralnet with 5 input_nodes and 1 output_node.
# short_term_memory=False disables recurrent flow in the network.
# set the name to 'docs/xor' (defaults to 'timestamp').
nn = pn.Neuralnet(5, 1, short_term_memory=False, name='docs/xor')

# set mutationrate to 1% (defaults to 5%),
nn.mutation_rate = 0.01

# load its weights (we already have a json dump. after each good mutation,
# the weights will be dumped into a json).
# this step is not neccesary.
#nn.load(nn.name)

# a dataset. this one yields XOR.
inputs = [[1,1,0,0,0], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1], [1,0,0,0,1], [0,0,0,1,1], [0,1,0,0,0], [1,1,1,1,1], [1,0,0,1,0]]
outputs = [[0], [1], [1], [1], [0], [0], [1], [0], [0]]

# define a fitness function to train with.
# any positively growing function with respect to
# positively counting attributes will suffice.
def fitness(): return -pn.MSE(nn, inputs, outputs)

# mount the fitness function onto the model.
nn.fitness_func = fitness

# train the model and show plots.
nn.train(10000, plot=True)

# use the model.
print('0 xor 0 xor 0 xor 1 xor 0 <=> ', nn.forward([0,0,0,1,0]))
```