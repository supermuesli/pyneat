from copy import deepcopy
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
		self.activation_func = sigmoid

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
	def __init__(self, input_layer_size, output_layer_size, mutation_rate=0.05, short_term_memory=False, recurrent_flow=False, decision_only=True, name=''):
		self.input_layer_size = input_layer_size
		self.output_layer_size = output_layer_size

		# use the name for convenient loading/saving of entire neuralnets
		self.name = name

		# all existing nodes
		self.nodes = [Node([], []) for i in range(output_layer_size + input_layer_size)]

		# to specificly target input and output nodes we introduce indices.
		self.o_start = 0
		self.o_end   = output_layer_size
		self.i_start = output_layer_size
		self.i_end   = output_layer_size+input_layer_size
		
		# use these to determine good/bad mutations.
		self.fitness = -math.inf
		self.fitness_func = None

		# number in range [0, 1]. probability of arbitrary mutation.
		self.mutation_rate = mutation_rate

		# enables short-term-memory of nodes if True
		self.short_term_memory = short_term_memory

		# enables recurrent flow if True.
		self.recurrent_flow = recurrent_flow

		# only allows for decision-activation-functions to be used
		self.decision_only = decision_only

		# for optimizing the mutate function a little
		self.has_adjacents = False

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
			node.value = node.activation_func(node.value)

		# saves values of last layer
		res = [node.value for node in self.nodes[self.o_start:self.o_end]]

		# reset all nodes based on their reset_rates.
		self.reset()

		return res

	def mutate(self):
		if self.has_adjacents:
			mutations = ['add_node', 'add_edge', 'weights']
			if self.short_term_memory:
				mutations += ['reset_rates']

			if not self.decision_only:
				mutations += ['activation_func']

			mutation_choice = mutations[int(next_float()*len(mutations))]
		else:
			mutation_choice = 'add_edge'

		# =====================================================================
		# mutate topology
		# =====================================================================
		
		if mutation_choice == 'add_node':
			from_to_indices = [int(next_float()*len(self.nodes)) for i in range(2)]

			if ((from_to_indices[0] < self.o_end) or (self.o_end <= from_to_indices[1] < self.i_end)) and (not (self.recurrent_flow or self.short_term_memory)):
				return 'no_mutation'

			from_to = [self.nodes[index] for index in from_to_indices]
			new_node = Node([from_to[1]], [1.0])
			self.nodes += [new_node]
			if new_node not in from_to[0].adjacents:
				from_to[0].adjacents += [new_node]
				from_to[0].weights += [next_neg_float()]
				return mutation_choice
			else:
				return 'no_mutation'

		elif mutation_choice == 'add_edge':
			from_to_indices = [int(next_float()*len(self.nodes)) for i in range(2)]

			if ((from_to_indices[0] < self.o_end) or (self.o_end <= from_to_indices[1] < self.i_end)) and (not (self.recurrent_flow or self.short_term_memory)):
				return 'no_mutation'

			from_to = [self.nodes[index] for index in from_to_indices]
			if from_to[1] not in from_to[0].adjacents:
				from_to[0].adjacents += [from_to[1]]
				from_to[0].weights += [next_neg_float()]
				self.has_adjacents = True
				return mutation_choice
			else:
				return 'no_mutation'
		# =====================================================================
		# mutate weights, reset_rates and activation_func
		# =====================================================================

		elif mutation_choice == 'weights':
			for node in self.nodes:
				# mutate weights
				for i in range(len(node.weights)):
					if next_float() < self.mutation_rate:
						node.weights[i] = next_neg_float()
			return mutation_choice

		elif mutation_choice == 'reset_rates':	
			# mutate reset_rate
			for node in self.nodes:
				if next_float() < self.mutation_rate:
					node.reset_rate = next_float()
			return mutation_choice

		elif mutation_choice == 'activation_func':
			# mutate activation_func
			for node in self.nodes:
				if next_float() < self.mutation_rate:
					node.activation_func = act_func_pool[next_float()*act_func_pool_len]
			return mutation_choice
			
	# create a visual graph of the neuralnet and open it
	def show(self):
		dot = Digraph(comment=self.name, graph_attr={'rankdir': 'LR'})
		dot.format = 'svg'

		for i in range(self.o_start, self.o_end):
			dot.node(str(self.nodes[i]), 'output_'+str(i), group='output', style='filled', fillcolor='#d62728', shape='rarrow', rank='sink')

		for i in range(self.i_start, self.i_end):
			dot.node(str(self.nodes[i]), 'input_'+str(i), group='input', style='filled', fillcolor='#1f77b4', shape='rarrow', rank='source')

		for i in range(self.i_end, len(self.nodes)):
			dot.node(str(self.nodes[i]), '', group='hidden', style='filled', fillcolor='#2ca02c', rank='same')

		for node in self.nodes:
			for i in range(len(node.adjacents)):
				dot.edge(str(node), str(node.adjacents[i]), label=str(round(node.weights[i], 6)))

		dot.render(filename=cwd + self.name + '_model')
		# TODO: refresh tab instead of opening new one each time.
		# https://stackoverflow.com/questions/16399355/refresh-a-local-web-page-using-python
		webbrowser.open('file://' + cwd + self.name + '_model.svg', new=0)

	# train for n cycles and save the topology/weights if they improved after each cycle.
	def train(self, cycles, plot=False, show=False):
		if cycles <= 0:
			print('ERROR: training cycles <= 0.')
			return

		if self.fitness_func == None:
			print('ERROR: fitness_func of model ', self.name, ' is None.')
			return

		test_fitness = self.fitness_func()
		accepted_types = [int, float, np.int64, np.float64]
		if type(test_fitness) not in accepted_types:
			print('ERROR: fitness_func of return-type ', type(test_fitness), ' of model ', self.name, ' is not in ', accepted_types)
			print('Either expand the accepted-types-list or fix your return-type.')
			return

		# fucking hell.
		# https://stackoverflow.com/questions/19210971/python-prevent-copying-object-as-reference
		prev_nodes = deepcopy(self.nodes)

		if plot:
			# used for fitness plotting at the end
			cycles_ = [i for i in range(1, cycles+1)]
			fitnesses_ = []
			for n in range(cycles):
				if self.mutate() == 'no_mutation':
					cycles_ = cycles_[:-1]
					continue
				fitness = self.fitness_func()

				# an improvement of at least 5% is required
				if fitness > self.fitness * 0.95:
					# good mutation. keep it.
					self.fitness = fitness		
					prev_nodes = deepcopy(self.nodes)
					self.save(self.name)

					if show:
						# show neuralnet in browser
						self.show()
				else:
					# bad mutation. revert.
					self.nodes = deepcopy(prev_nodes)
					
				fitnesses_ += [self.fitness]
		else:
			for n in range(cycles):
				if self.mutate() == 'no_mutation':
					continue
				fitness = self.fitness_func()

				if fitness > self.fitness:
					# good mutation. keep it.
					self.fitness = fitness		
					prev_nodes = deepcopy(self.nodes)
					self.save(self.name)

					if show:
						# show neuralnet in browser
						self.show()
				else:
					# bad mutation. revert.
					self.nodes = deepcopy(prev_nodes)
					
		if plot and fitnesses_ != []:
			# plot fitness development
			plt.clf()
			plt.figure(figsize=(10, 5))
			plt.xlabel('Training cycle', size=20)
			plt.ylabel('Fitness', size=20)
			plt.axis([0, cycles, min(fitnesses_) - 10, max(fitnesses_) + 10], size=15)
			plt.plot(cycles_, fitnesses_, color='red', linewidth=3)
			plt.grid()
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
		# an ID to each node. output and input nodes are going
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

			activation_func_str = 'sigmoid'
			if str(self.fitness_func).split('function ')[1][:-1] == 'linear':
				activation_func_str = 'linear'
			elif str(self.fitness_func).split('function ')[1][:-1] == 'relu':
				activation_func_str = 'relu'

			w_dict['node_'+str(i)] = {
				'adjacents': adjacents_, 
				'weights': self.nodes[i].weights,
				'reset_rate': self.nodes[i].reset_rate,
				'activation_func': activation_func_str
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
			new_node.activation_func = eval(w_dict['node_'+str(i-1)]['activation_func'])
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

	# reinitialize model while keeping the fitness_function.
	def forget(self):
		fitness_func_ = self.fitness_func
		self.__init__(self.input_layer_size, self.output_layer_size, self.mutation_rate, self.short_term_memory, self.recurrent_flow, self.decision_only, self.name)
		self.fitness_func = fitness_func_

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
	# set mutationrate to 20% (defaults to 5%),
	nn = Neuralnet(2, 2, mutation_rate=0.2, short_term_memory=False, recurrent_flow=False, decision_only=True, name='docs/2xor')

	# load the model (we already have a json dump. after each good mutation,
	# the model will be dumped into a json). this step is not neccesary.
	nn.load(nn.name)

	# a dataset. this one yields XOR.
	inputs = [[1,1], [1,0], [0,1], [0,0]]
	outputs = [[0,1], [1,0], [1,0], [0,1]]

	# define a fitness function to train with.
	# any positively growing function with respect to
	# positively counting attributes will suffice.
	def fitness(): return -MSE(nn, inputs, outputs)

	# mount the fitness function onto the model.
	nn.fitness_func = fitness

	# train the model, plot the fitness and show the topology after each good mutation if show=True.
	nn.train(1000, plot=True, show=False)

	# use the model.
	print('0 xor 1 <=> ', nn.forward([0,1]))

	# show the topology of the model
	nn.show()

	# if we want to retrain the model, we can make it forget its weights and topology
	nn.forget()

# =============================================================================
# main
# =============================================================================

def main():
	demo()

if __name__ == '__main__':
	main()	