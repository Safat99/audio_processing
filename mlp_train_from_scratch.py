# this is the code of the tutorial series 
#of  "The Sound of AI" from youtube >>> an awesome series that I am learning
#chapter 8 >> training a neural network 
# implementing backpropagation and gradient descent from scratch



import numpy as np
from random import random 



#save the activations and the derivatives
#implement backpropagation
#implement gradient descent
#implement train
#train our networks with some dummy dataset
#make some predictions


class MLP:
	
	def __init__(self,num_inputs=3, num_hidden=[3, 5], num_outputs=2):
	
		self.num_inputs = num_inputs
		self.num_hidden = num_hidden
		self.num_outputs = num_outputs
		
		
		layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
		
		#initiate random weights
		self.weights =[]
		for i in range(len(layers)-1):
			w = np.random.rand(layers[i], layers[i+1])
			self.weights.append(w)
		
		
		#save activations per layer
		activations = []
		for i in range(len(layers)):
			a = np.zeros(layers[i])
			activations.append(a)
		self.activations = activations


		#save derivatives per layer
		derivatives = []
		for i in range(len(layers)-1):
			d = np.zeros((layers[i], layers[i+1]))
			derivatives.append(a)
		self.derivatives = derivatives
		
			
	def forward_propagate(self, inputs):
		'''Computes forward propagation
		Args:
			inputs (ndarray): Input signals
		Returns:
			activations (ndarray) : Output values
		'''
		 
		activations = inputs
		self.activations[0] = inputs
		
		#iterate through the network layers
		for i,w in enumerate(self.weights):
			#calculate net inputs>> calculate matrix multiplication between prev acti and weight matrix
			net_inputs = np.dot(activations, w)
			
			#calculate the activations
			activations = self._sigmoid(net_inputs)
			self.activations[i+1] = activations
		#return output activation function
		return activations
	
	
	def back_propagate(self,error, verbose = False):
	
		# dE/dW_i = (y - a_[i+1] s'(h_[i+1])) a_i
		# s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
		#s(h_[i+1]) = a_[i+1]
		
		#dE/dW_[i-1] = (y - a_[i+1] s'(h_[i+1])) W_i s'(h_i) a_[i-1]
		
		for i in reversed(range(len(self.derivatives))):
			activations = self.activations[i+1]
			delta = error * self._sigmoid_derivative(activations)
			delta_reshaped = delta.reshape(delta.shape[0] , -1).T
			current_activations = self.activations[i]
			current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
			self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
			error = np.dot(delta, self.weights[i].T)
			
			
			if verbose:
				print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
	
		return error
		
	def gradient_descent(self, learning_rate):
		for i in range(len(self.weights)):
			weights = self.weights[i]
			#print("Original W{} {}".format(i,weights))
			
			derivatives = self.derivatives[i]
			
			weights += derivatives * learning_rate			
			#print("Updated W{} {}".format(i,weights))
	
	def train(self,inputs, targets, epochs, learning_rate):
		
		for i in range(epochs):
			sum_error = 0
			#for j,(input, target) in enumerate(zip(inputs, targets)):
			for input,target in zip(inputs,targets):
				#perform forward prop
				output = self.forward_propagate(input)
				
				#calculate the error
				error = target - output
				
				#back propagation
				self.back_propagate(error)
				
				#apply gradient descent
				self.gradient_descent(learning_rate)
				
				sum_error+=self._mse(target,output)
				
			#report error
			print("Error: {} at epoch {}".format(sum_error / len(inputs), i))	
	
	def _mse(self, target, output):
	#Mean Squared Error loss funciton
	'''Args:
		target,output>> ndarray >> The ground trut(?), the predicted values
	Returns:
		float>output
	'''
		return np.average((target - output)**2)
	
	def _sigmoid_derivative(self, x):
		return x * (1.0 - x)
	
	def _sigmoid(self, x):
		return 1/(1 + np.exp(-x))


		
if __name__ == "__main__":
	
	#create a dataset to train a network for the sum operation
	inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])#array([[0.1],[0.2], [0.3,0.4]])
	targets = np.array([[i[0] + i[1]] for i in inputs]) # array([[0.3],[0.7]])
	
	# create an MLP
	mlp = MLP(2, [5], 1)
	
	#train out mlp
	mlp.train(inputs, targets, 50, 0.1)
	
	
	
	
	#create dummy data
	input = np.array([0.3, 0.1])
	target = np.array([0.4])
	
	output = mlp.forward_propagate(input)
	print()
	print()
	print()
	print("Our network believes that {} + {} equal to {}".format(input[0], input[1], output[0]))
		
	'''
	#perform forward prop
	output = mlp.forward_propagate(input)
	
	#calculate the error
	error = target - output
	
	#back propagation
	mlp.back_propagate(error)
	
	#apply gradient descent
	mlp.gradient_descent(learning_rate = 1)
	'''
	
	
