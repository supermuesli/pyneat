import json, datetime, math, random, os, re, subprocess

# =============================================================================
# misc
# =============================================================================

# current working directory
cwd = os.getcwd() + '/'

# generate some floats as means of optimization.
random_floats = [random.random() for i in range(100000)]
random_floats_index = -1
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

# sigmoid function.
def sigmoid(x):
	return 1/(1 + math.exp(-x))

# rectified linear unit.
def relu(x):
	return max(0, x)

# linear unit.
def linear(x):
	return x

act_func_pool = [sigmoid, relu, linear]

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
	def __init__(self, adjacents, weights, activation_func, value=0):
		self.adjacents = adjacents
		self.weights = weights
		self.value = value
		self.activation_func = activation_func

	def forward(self):
		if len(self.adjacents) != len(self.weights):
			print(self.adjacents, self.weights)
			#print('ERROR: len(Node.adjacents != len(Node.weights))')
			return

		if self.adjacents == []:
			return
		for i in range(len(self.adjacents)):
			self.adjacents[i].value += self.value * self.weights[i]
			
class Neuralnet:
	def __init__(self, init_layer_sizes, activation_func, name=''):
		self.name = name
		self.activation_func = activation_func

		# first create empty nodes for all layers
		self.layers = [[Node([], [], self.activation_func) for j in range(init_layer_sizes[i])] for i in range(len(init_layer_sizes))]

		# now define the adjacent-lists for each node for each layer except the last.
		# to begin with, this will be a feed forward neural network.
		for i in range(len(self.layers)-1):
			for j in range(len(self.layers[i])):
				self.layers[i][j].adjacents = self.layers[i+1]
				self.layers[i][j].weights = [next_neg_float() for i in range(len(self.layers[i][j].adjacents))]

		# use this to undo a bad mutation.
		self.prev_layers = self.layers
		self.fitness = -math.inf

		# number in range [0, 1]. probability of weight mutation 
		self.mutation_rate = 0.05

		self.fitness_func = None

	# reset all node values back to 0.
	def reset(self):
		for layer in self.layers:
			for node in layer:
				node.value = 0

	# propagate forwards and return the resulting output layer as a list.
	# input_ is the list of values that are to be passed into the input layer.
	def forward(self, input_):
		if self.activation_func == None:
			print('ERROR: activation_func of model ', self.name, ' cannot be None.')
			return

		# assign input_ values to each node of the first layer
		for i in range(len(self.layers[0])):
			self.layers[0][i].value = input_[i]

		# forward
		for layer in self.layers:
			for node in layer:
				node.forward()

		# apply acitvation function per node
		for layer in self.layers:
			for node in layer:
				node.value = node.activation_func(node.value)

		# saves values of last layer
		res = [node.value for node in self.layers[-1]]

		# reset nodes
		self.reset()

		return res

	def mutate(self):
		# =====================================================================
		# mutate topology
		# =====================================================================
		# case 1: change amount of nodes of a layer
		#if next_float() < self.mutation_rate:
		#	# do not change input or output layers
		#	for i in range(1, len(self.layers)-1):
		#		# delete a node
		#		if next_float() < self.mutation_rate and sum([len(layer) for layer in self.layers]) > 1:
		#			# pick random node
		#			loser = int(next_float()*len(self.layers[i]))
		#			
		#			# delete its entries of incident lying nodes
		#			for j in range(len(self.layers)):
		#				for k in range(len(self.layers[j])):
		#					deletion_indices = [l for l, x in enumerate(self.layers[j][k].adjacents) if x == self.layers[i][loser]]
		#					for m in range(len(deletion_indices)): 
		#						del self.layers[i][loser].adjacents[deletion_indices[m]-m]
		#						del self.layers[i][loser].weights[deletion_indices[m]-m]
		#		
		#		# create a node
		#		if next_float() < self.mutation_rate:
		#			winners = []
		#			for layer_ in self.layers[i:]:
		#				for node_ in layer_:
		#					if next_float() < self.mutation_rate:
		#						winners += [node_]
		#			if winners != []:
		#				self.layers[i] += [Node(winners, [next_neg_float() for j in winners], act_func_pool[int(next_float()*len(act_func_pool))])]

		# =====================================================================
		# mutate weights
		# =====================================================================
		for layer in self.layers:
			for node in layer:
				for i in range(len(node.weights)):
					if next_float() < self.mutation_rate:
						node.weights[i] = next_neg_float()

		# =====================================================================
		# mutate activation functions
		# =====================================================================
		if next_float() < self.mutation_rate:
			pass

	# train for n cycles and save the topology/weights if they improved after each cycle.
	def train(self, cycles):
		if cycles <= 0:
			print('ERROR: training cycles <= 0.')
			return

		if self.fitness_func == None:
			print('ERROR: fitness_func of model ', self.name, ' is None.')
			return

		# TODO: multitreading
		for n in range(cycles):
			self.mutate()
			
			fitness = self.fitness_func()

			if fitness > self.fitness:
				# good mutation. keep it.
				self.fitness = fitness		
				self.prev_layers = self.layers
				self.save(self.name)		
			else:
				# bad mutation. revert.
				self.layers = self.prev_layers

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
		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				w_dict['layer_'+str(i)+'_node_'+str(j)] = self.layers[i][j].weights

		file_ = open(cwd + name + '.json', 'w')
		json.dump(w_dict, file_, indent=4, separators=(',', ': '), sort_keys=True)
		file_.close()

	# load weights from a json.
	def load(self, json_file):
		try:
			weights_file = open(cwd + json_file + '.json', 'r')
		except:
			print('ERROR: file ', json_file + '.json', ' not found.')
			return

		w_dict = json.load(weights_file)
		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				self.layers[i][j].weights = w_dict['layer_'+str(i)+'_node_'+str(j)]

		self.fitness = w_dict['fitness']
		self.prev_layers = self.layers

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
	# an initial activation function for all nodes
	# name 'docs/addition' (defaults to 'timestamp').
	nn = Neuralnet([2, 3, 4, 1], linear, 'docs/addition')
	
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
	def fitness(): return -MSE(nn, inputs, outputs)

	# mount the fitness function onto the model.
	nn.fitness_func = fitness

	# train the model.
	nn.train(1000000)

	# use the model.
	print('15 + 5 = ', nn.forward([15, 5]))


# =============================================================================
# main
# =============================================================================

def main():
	demo()

if __name__ == '__main__':
	main()	