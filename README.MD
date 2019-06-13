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

# a neuralnet with 2 input_nodes, 2 hidden_nodes, 1 output_node,
# an initial activation function for all nodes
# name 'docs/addition' (defaults to 'timestamp').
nn = Neuralnet([2, 1], pn.linear, 'docs/addition')

# set mutationrate to 10% (defaults to 5%),
nn.mutation_rate = 0.1

# load its weights (we already have a json dump. after each good mutation,
# the weights will be dumped into a json).
# this step is not neccesary.
nn.load(nn.name)

# a dataset. this one yields basic addition.
inputs = [[1,1], [2,2], [3,3], [4,4], [5,5], [1,0], [0.5,0.5], [10,100], [100,10]]
outputs = [[2], [4], [6], [8], [10], [1], [1], [110], [110]]

# define a fitness function to train with.
# any positively growing function with respect to
# positively counting attributes will suffice.
def fitness(): return -pn.MSE(nn, inputs, outputs)

# mount the fitness function onto the model.
nn.fitness_func = fitness

# train the model.
nn.train(1000)

# use the model.
print('15 + 5 = ', nn.forward([15, 5]))
```