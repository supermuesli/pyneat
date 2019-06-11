import numpy as np
import json, datetime, math, random, os

# =============================================================================
# misc
# =============================================================================

# current working directory
cwd = os.getcwd() + '/'

# generate some floats as means of optimization.
random_floats = [random.random() for i in range(100000)]
random_floats_index = -1
random_floats_length = len(random_floats)

def next_float():
	global random_floats_index

	random_floats_index = (random_floats_index + 1) % random_floats_length
	return random_floats[random_floats_index]

# =============================================================================
# activation functions
# =============================================================================

# sigmoid function.
def sigmoid(x):
	return 1/(1 + math.exp(-x))

# rectified linear unit.
def relu(x):
	return max(0, x)

# linear unit.
def linear(x):
	return x

# =============================================================================
# error functions
# =============================================================================

# returns squared error of prediction and expectation.
def SE(predicted_output, expected_output):
	return (predicted_output - expected_output)**2

# computes the mean squared error of the given neuralnet, inputs and outputs.
def MSE(neuralnet_, inputs, outputs):
	inputs_length = len(inputs)

	if inputs_length != len(outputs): 
		print('ERROR: len(inputs) != len(outputs).')
		return

	# summed error of this cycle
	summed_error = 0

	# forward propagate all inputs
	for i in range(inputs_length):
		output = neuralnet_.forward(inputs[i])
		
		# accumulate squared errors
		for j in range(len(outputs[i])):
			summed_error += SE(output[j], outputs[i][j])

	# return mean squared error
	return summed_error/inputs_length

# =============================================================================
# neural network implementation
# =============================================================================

class Neuralnet:
	def __init__(self, layers, activation_funcs, name=''):
		self.name = name
		self.layers = layers
		self.activation_funcs = activation_funcs

		# for instance [prev layer (1x5)] * [weight matrix (5x6)] -> [next layer (1x6)]
		self.weights = [np.zeros((layers[i], layers[i+1])) for i in range(len(layers)-1)]

		# use this to undo a bad mutation.
		self.prev_weights = [np.zeros((layers[i], layers[i+1])) for i in range(len(layers)-1)]
		self.fitness = -math.inf 

		# number in range [0, 1]. probability of weight mutation 
		self.mutation_rate = 0.05

		self.fitness_func = None

	# propagate forwards and return the resulting output layer as a list.
	# input_ is the list of values that are to be passed into the input layer.
	def forward(self, input_):
		if len(self.activation_funcs) != len(self.layers) -1:
			print('ERROR: amount of activation functions != amount of layers - 1')
			return

		layer = [np.zeros((1, self.layers[i])) for i in range(len(self.layers))]
		layer[0] = np.array(input_)
		for i in range(0, len(self.layers)-1):
			layer[i+1] = [self.activation_funcs[i](j) for j in np.dot(layer[i], self.weights[i])]

		return list(layer[-1])

	# (uniform)-randomly mutate weights - simplest possible gene-mutation :).
	# TODO: importance-mutation
	def mutate(self):
		for i in range(len(self.weights)):
			# self.weights[i] is a matrix
			for j in range(len(self.weights[i])):
				# self.weights[i][j] is a row
				for k in range(len(self.weights[i][j])):
					# self.weights[i][j] is a weight (float)
					if next_float() < self.mutation_rate:
						self.weights[i][j][k] = next_float() if next_float() < 0.5 else -next_float()
		
		self.weights = np.array(self.weights)

	# train for n cycles and save the weights if they improved after each cycle.
	def train(self, cycles):
		if cycles <= 0:
			print('ERROR: training cycles <= 0.')
			return

		if self.fitness_func == None:
			print('ERROR: Neuralnet.fitness_func = None.')
			return

		for n in range(cycles):
			self.mutate()
			
			fitness = self.fitness_func()

			if fitness >= self.fitness:
				# good mutation. keep it.
				self.fitness = fitness		
				self.prev_weights = self.weights
				self.save(self.name)		
			else:
				# bad mutation. revert.
				self.weights = self.prev_weights

		# i don't know what's wrong with the training-loop
		# but if i don't load the most recently saved weights
		# then the weights of the neuralnet are random shit.
		self.load(self.name)

		print('fitness: ', self.fitness)

	# dump the weights into a json. 
	# you may provide an output name (defaults to datetime).
	def save(self, name=''):
		if name == '':
			date_time = (str(datetime.datetime.now()).split('.')[0]).split(' ')
			name = date_time[0] + '_' + date_time[1] + '_weights'
		
		w_dict = {'fitness': self.fitness}
		for i in range(len(self.weights)):
			w_dict['weights_layer_'+str(i)] = [list(map(lambda x: float(x), row)) for row in self.weights[i]]

		file_ = open(cwd + name + '.json', 'w')
		json.dump(w_dict, file_, indent=4, separators=(',', ': '), sort_keys=True)
		file_.close()

	# load weights from a json.
	def load(self, json_file):
		try:
			weights_file = open(cwd + json_file + '.json', 'r')
		except:
			print('WARNING: file ', json_file + '.json', ' not found. New file will be created.')
			weights_file = open(cwd + json_file + '.json', 'w')

		w_dict = json.load(weights_file)
		for i in range(len(self.weights)):
			self.weights[i] = np.array(w_dict['weights_layer_'+str(i)])

		self.fitness = w_dict['fitness']
		self.prev_weights = self.weights

# =============================================================================
# demo
# =============================================================================

def demo():
	# allows you to use these out of scope.
	global nn
	global inputs
	global outputs

	# this demo implements a neural network that learns a dataset by applying NEAT.

	# a neuralnet with 2 input_nodes, 3 hidden_nodes, 4 hidden_nodes, 1 output_node,
	# 3 (amount of layers - 1) linear activation functions,
	# name 'addition' (defaults to 'timestamp').
	nn = Neuralnet((2, 3, 4, 1), (linear, linear, linear), 'addition')
	
	# set mutationrate to 10% (defaults to 5%),
	nn.mutation_rate = 0.1

	# load its weights (we already have a json dump. after each good mutation,
	# the weights will be dumped into a json).
	# this step is not neccesary.
	nn.load('docs/' + nn.name)

	# a dataset. this one yields basic addition.
	inputs = [[1,1], [2,2], [3,3], [4,4], [5,5], [1,0], [0.5,0.5], [10,100], [100,10]]
	outputs = [[2], [4], [6], [8], [10], [1], [1], [110], [110]]

	# define a fitness function to train with.
	# any positively growing function with respect to
	# positively counting attributes will suffice.
	def fitness(): return -MSE(nn, inputs, outputs)

	# mount the fitness function onto the model.
	nn.fitness_func = fitness

	# train the model.
	nn.train(1000)

	# use the model.
	print('15 + 5 = ', nn.forward([15, 5]))


# =============================================================================
# main
# =============================================================================

def main():
	demo()

if __name__ == '__main__':
	main()	