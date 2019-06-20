# pyneat

A python module that implements neural networks capable of applying NEAT.

#### python 3.7.1+ dependencies
- numpy
- graphviz
- matplotlib
- ujson
- cython

#### software dependencies
- graphviz

#### how to build
simply run

	$ make firstBuild

#### what to expect of pyneat
- generate arbitrary neural networks made for reinforcement learning
- save and load models using json dumps
- easily enable/disable recurrent flow, short-term-memory and decision-only-state

#### usage
```python
# this demo implements a neural network that learns a dataset by applying NEAT.

import pyneat as pn

# a neuralnet with 5 input_nodes and 1 output_node.
# short_term_memory=False disables recurrent flow in the network.
# set the name to 'docs/xor' (defaults to 'timestamp').
# set mutationrate to 20% (defaults to 5%),
nn = pn.Neuralnet(2, 2, mutation_rate=0.2, short_term_memory=False, recurrent_flow=False, decision_only=True, name='docs/2xor')

# load the model (we already have a json dump. after each good mutation,
# the model will be dumped into a json). this step is not neccesary.
nn.load(nn.name)

# a dataset. this one yields XOR.
inputs = [[1,1], [1,0], [0,1], [0,0]]
outputs = [[0,1],[1,0],[1,0],[0,1]]       

# define a fitness function to train with.
# any positively growing function with respect to
# positively counting attributes will suffice.
def fitness(): return -pn.MSE(nn, inputs, outputs)

# mount the fitness function onto the model.
nn.fitness_func = fitness

# train the model for 1000 cycles, plot the fitness (if plot=True) and show the topology after each good mutation if (show=True).
nn.train(1000, plot=True, show=False)

# use the model.
print('0 xor 1 <=> ', nn.forward([0,1]))

# manually show the topology of the model
nn.show()

# if we want to retrain the model, we can make it forget its weights and topology
nn.forget()
```

#### showcase
