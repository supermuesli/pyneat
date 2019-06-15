from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt
import ujson as json
import datetime, math, random, os, webbrowser, re, subprocess

# =============================================================================
# misc
# =============================================================================

# current working directory
cwd = os.getcwd() + '/'

# generate some floats as means of optimization.
random_floats = [random.random() for i in range(100000)]
random_floats_index = 0
random_floats_length = len(random_floats)

# returns the next random float in sequence.
def next_float():
	global random_floats_index

	random_floats_index = (random_floats_index + 1) % random_floats_length
	return random_floats[random_floats_index]

def next_neg_float():
	return next_float() if next_float() < 0.5 else -next_float()

# returns the number of available cpu cores.
def cpu_count():
	""" Number of available virtual or physical CPUs on this system, i.e.
	user/real as output by time(1) when called with an optimally scaling
	userspace-only program"""

	# cpuset
	# cpuset may restrict the number of *available* processors
	try:
		m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
					  open('/proc/self/status').read())
		if m:
			res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
			if res > 0:
				return res
	except IOError:
		pass

	# Python 2.6+
	try:
		import multiprocessing
		return multiprocessing.cpu_count()
	except (ImportError, NotImplementedError):
		pass

	# https://github.com/giampaolo/psutil
	try:
		import psutil
		return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
	except (ImportError, AttributeError):
		pass

	# POSIX
	try:
		res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

		if res > 0:
			return res
	except (AttributeError, ValueError):
		pass

	# Windows
	try:
		res = int(os.environ['NUMBER_OF_PROCESSORS'])

		if res > 0:
			return res
	except (KeyError, ValueError):
		pass

	# jython
	try:
		from java.lang import Runtime
		runtime = Runtime.getRuntime()
		res = runtime.availableProcessors()
		if res > 0:
			return res
	except ImportError:
		pass

	# BSD
	try:
		sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
								  stdout=subprocess.PIPE)
		scStdout = sysctl.communicate()[0]
		res = int(scStdout)

		if res > 0:
			return res
	except (OSError, ValueError):
		pass

	# Linux
	try:
		res = open('/proc/cpuinfo').read().count('processor\t:')

		if res > 0:
			return res
	except IOError:
		pass

	# Solaris
	try:
		pseudoDevices = os.listdir('/devices/pseudo/')
		res = 0
		for pd in pseudoDevices:
			if re.match(r'^cpuid@[0-9]+$', pd):
				res += 1

		if res > 0:
			return res
	except OSError:
		pass

	# Other UNIXes (heuristic)
	try:
		try:
			dmesg = open('/var/run/dmesg.boot').read()
		except IOError:
			dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
			dmesg = dmesgProcess.communicate()[0]

		res = 0
		while '\ncpu' + str(res) + ':' in dmesg:
			res += 1

		if res > 0:
			return res
	except OSError:
		pass

	raise Exception('Can not determine number of CPUs on this system')

# =============================================================================
# activation functions
# =============================================================================

# optimized sigmoid function.
def sigmoid(x):
	# prevent overflow.
	x = np.clip(x, -50, 50)

	return 1.0/( 1 + np.exp(-4.9*x))

# rectified linear unit.
def relu(x):
	return max(0, x)

# linear unit.
def linear(x):
	return x

# activation functions pool.
act_func_pool = [sigmoid, relu, linear]
act_func_pool_len = len(act_func_pool)

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

class Node:
	def __init__(self, adjacents, weights):
		self.adjacents = adjacents
		self.weights = weights
		self.value = 0

		# number in range [0, 1]. percentage of a nodes value that will remain 
		# when the neuralnet is reset (initially 0). this will sort of mimic 
		# memory. reset_rates can mutate per node. 
		self.reset_rate = 0

	def forward(self):
		if len(self.adjacents) != len(self.weights):
			print('ERROR: len(Node.adjacents != len(Node.weights))')
			return

		if self.adjacents == []:
			return

		for i in range(len(self.adjacents)):
			self.adjacents[i].value += self.value * self.weights[i]

	def reset(self):
		self.value *= self.reset_rate
			
class Neuralnet:
	def __init__(self, input_layer_size, output_layer_size, short_term_memory=False, name=''):
		# use the name for convenient loading/saving of entire neuralnets
		self.name = name

		# all existing nodes
		self.nodes = [Node([], []) for i in range(output_layer_size + input_layer_size)]

		# never add adjacent nodes onto output nodes. other mutations are fine.
		# to specificly target input and output nodes we introduce indices.
		self.o_start = 0
		self.o_end   = output_layer_size
		self.i_start = output_layer_size
		self.i_end   = output_layer_size+input_layer_size
		
		# use this to undo a bad mutation.
		self.prev_nodes = self.nodes
		
		# use these to determine good/bad mutations.
		self.fitness = -math.inf
		self.fitness_func = None

		# number in range [0, 1]. probability of arbitrary mutation.
		self.mutation_rate = 0.05

		# basically enables recurrent flow if True.
		self.short_term_memory = short_term_memory

	# resets all node values based on their reset_rates.
	def reset(self):
		for node in self.nodes:
			node.reset()

	# propagate forwards and return the resulting output layer as a list.
	# input_ is the list of values that are to be passed into the input layer.
	def forward(self, input_):
		# assign input_ values to each node of the first layer
		for i, node in enumerate(self.nodes[self.i_start:self.i_end]):
			node.value = input_[i]

		# forward
		for node in self.nodes:
			node.forward()

		# apply acitvation function per node
		for node in self.nodes:
			node.value = sigmoid(node.value)

		# saves values of last layer
		res = [node.value for node in self.nodes[self.o_start:self.o_end]]

		# reset all nodes based on their reset_rates.
		self.reset()

		return res

	def mutate(self):
		mutations = ['add_node', 'add_edge', 'weights', 'reset_rates']
		mutation_choice = mutations[int(next_float()*len(mutations))]

		# =====================================================================
		# mutate topology
		# =====================================================================
		
		# add new node
		if mutation_choice == 'add_node':
			from_to = [self.nodes[int(next_float()*len(self.nodes))] for i in range(2)]
			new_node = Node([from_to[1]], [1])
			if from_to[0].adjacents != []:
				self.nodes += [new_node]
				from_to[0].adjacents = from_to[0].adjacents[:-1] + [new_node]
			return

		# add new edge
		if mutation_choice == 'add_edge':
			from_to = [self.nodes[int(next_float()*len(self.nodes))] for i in range(2)]
			if from_to[1] not in from_to[0].adjacents: 
				from_to[0].adjacents += [from_to[1]]
				from_to[0].weights += [next_neg_float()]
			return

		# =====================================================================
		# mutate weights and reset_rates
		# =====================================================================

		if mutation_choice == 'weights':
			for node in self.nodes:
				# mutate weights
				for i in range(len(node.weights)):
					if next_float() < self.mutation_rate:
						node.weights[i] = next_neg_float()
			return

		if mutation_choice == 'reset_rate':	
			# mutate reset_rate
			for node in self.nodes:
				if next_float() < self.mutation_rate:
					node.reset_rate = next_float()
			return
			
	# create a visual graph of the neuralnet and open it
	def show(self):
		dot = Digraph(comment=self.name, graph_attr={'rankdir': 'LR'})
		dot.format = 'svg'

		for i in range(self.o_start, self.o_end):
			dot.node(str(self.nodes[i]), 'output_'+str(i), style='filled', fillcolor='#d62728', shape='rarrow', rank='rightmost!')

		for i in range(self.i_start, self.i_end):
			dot.node(str(self.nodes[i]), 'input_'+str(i), style='filled', fillcolor='#1f77b4', shape='rarrow', rank='leftmost!')

		for i in range(self.i_end, len(self.nodes)):
			dot.node(str(self.nodes[i]), '', style='filled', fillcolor='#2ca02c')

		for node in self.nodes:
			for i in range(len(node.adjacents)):
				dot.edge(str(node), str(node.adjacents[i]), label=str(round(node.weights[i], 2)))

		dot.render(filename=cwd + self.name + '_model')
		webbrowser.open('file://' + cwd + self.name + '_model.svg')

	# train for n cycles and save the topology/weights if they improved after each cycle.
	def train(self, cycles, plot=False):
		if cycles <= 0:
			print('ERROR: training cycles <= 0.')
			return

		if self.fitness_func == None:
			print('ERROR: fitness_func of model ', self.name, ' is None.')
			return

		test_fitness = self.fitness_func()
		accepted_types = [int, float, np.int64, np.float64]
		if type(test_fitness) not in accepted_types:
			print('ERROR: fitness_func of of type ', type(test_fitness), ' of model ', self.name, ' is not in ', accepted_types)
			return

		if plot:
			# used for plotting at the end
			x_ = [i for i in range(1, cycles+1)]
			y_ = []

			for n in range(cycles):
				self.mutate()
				fitness = self.fitness_func()

				if fitness > self.fitness:
					# good mutation. keep it.
					self.fitness = fitness		
					self.prev_nodes = self.nodes
					self.save(self.name)		

					# show neuralnet in browser
					self.show()
				else:
					# bad mutation. revert.
					self.nodes = self.prev_nodes

				y_ += [self.fitness]
		else:
			for n in range(cycles):
				self.mutate()
				fitness = self.fitness_func()

				if fitness > self.fitness:
					# good mutation. keep it.
					self.fitness = fitness		
					self.prev_nodes = self.nodes
					self.save(self.name)		
				else:
					# bad mutation. revert.
					self.nodes = self.prev_nodes

		# i don't know what's wrong with the training-loop
		# but somehow i have to reload the weights
		self.load(self.name)

		if plot:
			# plot fitness development
			plt.clf()
			plt.xlabel('Training cycle')
			plt.ylabel('Fitness')
			plt.axis([0, cycles, min(y_) - 10, max(y_) + 10])
			plt.plot(x_, y_)
			plt.savefig(cwd+self.name+'_fitness.svg')

			# show fitness plot in browser
			webbrowser.open('file://' + cwd + self.name + '_fitness.svg')

		print('fitness: ', self.fitness)

	# dump the model into a json. 
	# you may provide an output name (defaults to timestamp).
	def save(self, name=''):
		if name == '':
			date_time = (str(datetime.datetime.now()).split('.')[0]).split(' ')
			name = date_time[0] + '_' + date_time[1] + '_weights'
		
		w_dict = {}
		w_dict['fitness'] = self.fitness

		# you can't dump memory adresses into a json, so assign 
		# an ID to each node. putput and input nodes are going
		# to be retrieved by using the initial_layer_sizes 
		# provided at the declaration of the neuralnet. this
		# works because output and input nodes are the first
		# nodes stored in Neuralnet.nodes .
		node_ids = {} 
		seen = []
		id_ = 0
		for node in self.nodes:
			if node not in seen:
				node_ids[node] = id_
				id_ += 1

		for i in range(len(self.nodes)):
			adjacents_ = []
			for key in node_ids:
				for j, adjacent in enumerate(self.nodes[i].adjacents):
					if adjacent == key:
						adjacents_ += [node_ids[key]]

			w_dict['node_'+str(i)] = {
				'adjacents': adjacents_, 
				'weights': self.nodes[i].weights,
				'reset_rate': self.nodes[i].reset_rate
			}

		file_ = open(cwd + name + '.json', 'w')
		json.dump(w_dict, file_, indent=4, sort_keys=True)
		file_.close()

	# load model from a json.
	def load(self, json_file):	
		try:
			model_file = open(cwd + json_file + '.json', 'r')
		except:
			print('ERROR: file ', json_file + '.json', ' not found.')
			return

		self.nodes = []
		w_dict = json.load(model_file)
		
		for i, key in enumerate(w_dict):
			if i == 0: 
				continue
			new_node = Node(w_dict['node_'+str(i-1)]['adjacents'], w_dict['node_'+str(i-1)]['weights'])
			new_node.reset_rate = w_dict['node_'+str(i-1)]['reset_rate']
			self.nodes += [new_node]

		# retrieve adresses
		adresses = {}
		for i in range(len(self.nodes)):
			adresses[i] = self.nodes[i]

		#replace them with the indices in Node.adjacents
		for node in self.nodes:
			for key in adresses:
				for j, adjacent in enumerate(node.adjacents):
					if adjacent == key:
						node.adjacents[j] = adresses[key]

		self.fitness = w_dict['fitness']
		self.prev_nodes = self.nodes

# =============================================================================
# demo
# =============================================================================

def demo():
	# allows you to use these out of scope.
	global nn
	global inputs
	global outputs

	# this demo implements a neural network that learns a dataset by applying NEAT.

	# a neuralnet with 5 input_nodes and 1 output_node.
	# short_term_memory=False disables recurrent flow in the network.
	# set the name to 'docs/xor' (defaults to 'timestamp').
	nn = Neuralnet(5, 1, short_term_memory=False, name='docs/xor')
	
	# set mutationrate to 1% (defaults to 5%),
	nn.mutation_rate = 0.01

	# load the model (we already have a json dump. after each good mutation,
	# the model will be dumped into a json).
	# this step is not neccesary.
	# nn.load(nn.name)

	# a dataset. this one yields XOR.
	inputs = [[1,1,0,0,0], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1], [1,0,0,0,1], [0,0,0,1,1], [0,1,0,0,0], [1,1,1,1,1], [1,0,0,1,0]]
	outputs = [[0], [1], [1], [1], [0], [0], [1], [0], [0]]

	# define a fitness function to train with.
	# any positively growing function with respect to
	# positively counting attributes will suffice.
	def fitness(): return -10*MSE(nn, inputs, outputs)

	# mount the fitness function onto the model.
	nn.fitness_func = fitness

	# train the model and show plots.
	nn.train(1000, plot=True)

	# use the model.
	print('0 xor 0 xor 0 xor 1 xor 0 <=> ', nn.forward([0,0,0,1,0]))


# =============================================================================
# main
# =============================================================================

def main():
	demo()

if __name__ == '__main__':
	main()	