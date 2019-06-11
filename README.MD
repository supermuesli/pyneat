# pyneat

#### python3 requirements
- numpy

#### what to expect of pyneat
- easily generate arbitrary neural networks made for reinforcement as well as supevised learning
- easily set activation functions per layer
- easily provide a fitness function per model
- easily save and load weights using json

#### usage
```python
# this demo implements a neural network that learns a dataset by applying NEAT.

import pyneat as pn

# create a neuralnet with 2 input_nodes, 3 hidden_nodes, 4 hidden_nodes and 1 output_node.
# specify the 3 (amount of layers - 1) activation functions, in this case all linear.
# set the name of thee model to 'addition' (defaults to 'timestamp').
nn = pn.Neuralnet((2, 3, 4, 1), (pn.linear, pn.linear, pn.linear), 'addition')

# set mutationrate to 10% (defaults to 5%),
nn.mutation_rate = 0.1

# load the models weights (we already have a json dump).
nn.load('docs/' + nn.name)

# a dataset. this one yields basic addition.
inputs = [[1,1], [2,2], [3,3], [4,4], [5,5], [1,0], [0.5,0.5], [10,100], [100,10]]
outputs = [[2], [4], [6], [8], [10], [1], [1], [110], [110]]

# define a fitness function to train with.
# any positively growing function with respect to
# positively counting attributes will suffice. pn. MSE computes the mean squared error
# of a supervised learning model.
def fitness(): return -pn.MSE(nn, inputs, outputs)

# mount the fitness function onto the model.
nn.fitness_func = fitness

# train the model. after each good mutation, the weights will be dumped into its json).
nn.train(1000)

# use the model.
print('15 + 5 = ', nn.forward([15, 5]))
```