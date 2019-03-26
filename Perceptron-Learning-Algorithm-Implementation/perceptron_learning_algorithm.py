import csv
import sys
import numpy as np 
import scipy as sp 
import pandas as pd 
import sklearn 
from sklearn.model_selection import train_test_split

def process_data(argv):
	'''
	Implements data processing using pandas and numpy

	Arguments:
	argv -- input file

	Returns:
	X_train -- input features for training of shape (# features, # examples)
	Y_train -- labels for trainin fo shape (# examples)
	X_test -- input features for test of shape (# featuers, # examples)
	Y_test -- true labels for test of shape (# examples)
	dim -- # features of input
	'''
	inputfile = argv
	print(inputfile[1])

	df = pd.read_csv(inputfile[1])																	# Read csv file using pandas

	X = df[['feature_1', 'feature_2']]																# Assign X and Y
	Y = df['label']

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=82)	# Test-train data split

	X_train = X_train.T # Transpose input data
	X_test = X_test.T

	Y_train = np.array(Y_train)									# Transform Y_train and Y_test to numpy array
	Y_train.astype(float)
	Y_test = np.array(Y_test)
	Y_test.astype(float)

	_, dim = X.shape 										# Get input dimensions

	return X_train, Y_train, X_test, Y_test, dim

def initialize_parameters(dim):
	'''
	Implements parameter initialization for Weights (W) and bias (b)

	Arguments:
	dim -- dimension of the input shape (2, 1)

	Returns: 
	W -- float, weight matix of shape (# features, 1) initialized randomly from normal distribution
	b -- bias parameter, float value of zero
	'''
	W = np.random.randn(dim,1)							# Initialize W and b
	b = 0

	assert(W.shape == (dim, 1))							# Check dimension with assert()
	assert(isinstance(b, float) or isinstance(b, int))

	return W, b

def hyperbolic(Z):
	'''
	Implements tanh non-linearization for non-linearization of perceptron model

	Arguments:
	Z -- linearized value of the perceptron unit of shape (1, # examples)

	Returns:
	A -- Non-linearized value of the perceptron unit of shape (1, # examples)
	'''
	A = ((np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z)))				# Calculate hyperbolic/tanh function

	return A

# Cost function
def compute_cost(A, Y):
	'''
	Implements Hinge loss using sklearn.metrics

	Arguments:
	A -- non-linearized output of the perceptron unit os shape (1, # examples)

	Returns:
	cost -- scalar value, cost of the output measured using the hinge loss equation, loss(y) = max(0, 1 - y_predicted * y_true)
	'''
	cost = sklearn.metrics.hinge_loss(Y, A)

	return cost

def propagate(X, Y, W, b):
	'''
	Implements forward and back propagation for the perceptron learning model

	Arguments:
	X -- input matrix, shape of (# features, # examples)
	Y -- true label, vector containing 1 or -1, of size (1, # examples)
	W -- weight matrix of shape (# features, 1)
	b -- bias parameter, float

	Returns:
	cost -- scalar value determined by the loss function 
	dW -- gradient of the loss with respect to w, shape (# features, 1)
	db -- gradient of the loss with respec to b, thus same shape as b
	'''

	m = X.shape[0]
	
	A = hyperbolic(np.dot(W.T, X) + b)					# Forward propagation (from X to cost)
	A = np.squeeze(A)
	cost = compute_cost(A, Y)
	
	dW = (1/m)*np.dot(X, (A - Y).T)						# Backward propagation (from cost to X)
	db = (1/m)*np.sum(A - Y)
	dW = np.reshape(dW, (2,1))

	assert(dW.shape == W.shape)						# Assert shape and type
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	grads = {'dW': dW,	
			'db': db}						# Dictionary to store derivatives

	return grads, cost 

def optimize(X, Y, W, b, num_iteration, learning_rate, print_cost):
	'''
	Implements the perceptron learning algorithm 
	Goal: learn W and b by minimizing the cost function 
	Update rule: weight = weight - learning_rate * gradient of weight

	Arguments:
	X -- training data of shape (# features, # examples)
	Y -- training label  data of shape (1, # examples)
	W -- wieght matix  of shape (# features, 1)
	b -- bias, a float
	num_iteration -- hyperparameter representing the number of iterations to optimize the parameters
	learning_rate -- hyperparameter representing the learning rate used in the optimize()
	print_cost -- set to true to print the cost every 100 iterations

	Returns:
	params -- dictionary containing the weights W and bias b
	grads -- dictionary containing the gradients of weights and bias with respect to cost function
	costs -- list of all the costs computed during the optimiation 
	'''
	costs = []

	for i in range(num_iteration):
		
		grads, cost = propagate(X, Y, W, b)			# Costs and gradient calculation

		dW = grads['dW']					# Retrieve derivatives from grads
		db = grads['db']

		W = W - learning_rate*dW				# Update rule
		b = b - learning_rate*db

		if i % 100 == 0:					# Record the costs
			costs.append(cost)

		if print_cost and i % 100 == 0:				# Print the cost every 100 training iterations
			print('Cost after iteration %i: %f' %(i, cost))

	params = {'W': W,
			'b': b}						# Dictionary for paramters

	grads = {'dW': dW,
			'db': db}					# Dictionary for gradients

	return params, grads, costs

def predict(X, W, b):
	'''
	Predicts the labels for the test examples

	Arguments:
	X -- test data of size (# features, # examples)
	W -- weights, a numpy array of size (# features, 1)
	b -- bias, a scalar

	Returns:
	Y_prediction -- a numpy array (vector) containing binary predictions (-1,1) for examples in X_test
	'''

	m = X.shape[1]
	Y_prediction = np.zeros((1,m))					# Initialize prediction array
	W = W.reshape(X.shape[0],1)					# Reshape weight matrix

	Y_prediction = hyperbolic(np.dot(W.T, X) + b)			# Predict Y_prediction probabilites

	Y_prediction[Y_prediction > 0] = 1				# Assign 1 -or -1 based on calculated probabilities
	Y_prediction[Y_prediction <= 0] = -1

	assert(Y_prediction.shape == (1,m))				# Assert shape

	return Y_prediction

def model(X_train, Y_train, X_test, Y_test, dim, num_iteration=400, learning_rate=0.01, print_cost=True):
	'''
	Implements the perceptron model by calling the helper function built above

	Arguments
	X and Y for the moment
	X_train -- training examples set represented by a numpy array of shape (# features, # training examples)
	Y_train -- training labels set represented by a numpy array of shape (1, # training examples)
	X_test -- test set represented by a numpy array of shape (# features, # test examples)
	Y_test -- test labels represented by a numpy array of shape (1, # test lables)
	num_iteration -- hyperparameter representing the number of iterations to optimize the parameters
	learning_rate -- hyperparameter representing the learning rate used in the optimize()
	print_cost -- set to true to print the cost every 100 iterations

	Returns:
	output -- dictionary containing information about the model
	'''

	W, b = initialize_parameters(dim)				# Initialize parameters
	print('initialized paramters!')

	print('Optimizing!')						# Gradient descent
	parameters, grads, costs = optimize(X_train, Y_train, W, b, num_iteration, learning_rate, print_cost)

	W = parameters['W']						# Retrieve parameters W and b from dictionary parameters
	b = parameters['b']

	Y_prediction_test = predict(X_test, W, b)			# Prediction test/train set examples
	Y_prediction_train = predict(X_train, W, b)

	print(Y_prediction_test)
	print(Y_test)
	print(Y_prediction_train)
	print(Y_train)

	print('Train accuracy: {}%'.format((np.sum(Y_prediction_train == Y_train)/Y_train.shape[0])*100))	# Print test/train errors
	print('Test accuracy: {}%'.format((np.sum(Y_prediction_test == Y_test)/Y_test.shape[0])*100))

	output = {'costs': costs,																		
				'Y_prediction_train': Y_prediction_train,
				'Y_prediction_test': Y_prediction_test,
				'W': W,
				'b': b,
				'learning_rate': learning_rate,
				'num_iteration': num_iteration}		# Output dictionary

	return output

def main(argv):

	# Hyperparamters
	num_iteration = 2000
	learning_rate = 0.005
	print_cost = True

	# Process data to get training and testing data
	X_train, Y_train, X_test, Y_test, dim = process_data(argv)

	print('Y train dim: ', Y_train.shape)
	print('Y test dim: ', Y_test.shape)

	# Build perceptron learning model
	output = model(X_train, Y_train, X_test, Y_test, dim, num_iteration=num_iteration, learning_rate=learning_rate, print_cost=print_cost)

	print('output: ', output['costs'])
	print('Parameters: ', output['W'], output['b'])

	W = output['W']
	b = output['b']
	list1 = [W[0,0], W[1,0], b]

	# Write output to a csv file - final weights and bias values 
	with open('output_1.csv', 'w', newline='') as myfile:
		wr = csv.writer(myfile, delimiter=',') 
		wr.writerow(list1)

if __name__=='__main__':
	main(sys.argv)

print('Program End!')
