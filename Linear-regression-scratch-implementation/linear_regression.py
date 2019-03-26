# Linear Regression Scratch Implementation

import sys
import csv
import numpy as np 
import pandas as pd 

# Read input file
def get_file(argv):
	'''
	Gets data file from input and converts it to pandas dataframe

	Arguments:
	argv -- input data file in csv format

	Returns:
	X -- pandas dataframe of feature values
	Y -- pandas dataframe of label values
	'''
	infile = argv									# get data file 		
	df = pd.read_csv(infile[1])							# read csv file using pandas
	print(df)
	df.rename(columns={'blood-fat-content':'bldfat'}, inplace=True)			# rename a column header
	print('Data statistical description: ')
	print(df.describe(), '\n')							# statistical description of data
	X = df[['b', 'weight', 'age']]							# assign X
	Y = df['bldfat']								# assign Y

	print(X)
	return X, Y

# Data normalizaton 
def normalize(X):
	'''
	Implements data normalization using mean and standard deviation scaling

	Arguments:
	X -- array of feature values 

	Returns:
	X -- array scaled features 
	'''

	X['weight'] = (X['weight'] - X['weight'].mean())/(X['weight'].std())
	X['age'] = (X['age'] - X['age'].mean())/(X['age'].std())
	
	return X

# Parameter initialization
def initialize_parameters():
	'''
	Implements parameter initialization using numpy random number generation from a normal distribution sample

	Returns:
	parameters -- coefficients of linear regression initialized form normal random distribution
	'''

	# np.random.seed(10)								# can use a seed to output consistent numbers
	beta_0 = np.random.randn()							# initialize parameters
	beta_1 = np.random.randn()
	beta_2 = np.random.randn()

	parameters = {'beta_0': beta_0,
				'beta_1': beta_1,
				'beta_2': beta_2}					# dictionary to store parameter values

	return parameters

# Compute prediction - y-hat
def predict_y(X, params):
	'''
	Computes predicted-y for input X with beta parameters

	Arguments:
	X -- array of feature values 
	parameters -- dictionary of beta parameters

	Returns:
	Y_predicted -- predicted Y-value of based on current parameters defined by betas
	'''

	beta_0 = params['beta_0']							# Retrieve beta values from dictionary
	beta_1 = params['beta_1']
	beta_2 = params['beta_2']

	Y_predicted = beta_0*X[0,:] + beta_1*X[1,:] + beta_2*X[2,:]			# Calculate y prediction
	
	return Y_predicted

# Cost function
def compute_cost(X, Y, params):
	'''
	Computes cost for the training data set

	Arguments:
	X -- array of feature values 
	Y -- array of label values
	params -- dictionary of paramters 

	Returns:
	grads -- dictionary of gradient 
	cost -- float value of cost calculated by least squares 
	'''
	db_0 = 0									# Initialize gradients to zero
	db_1 = 0
	db_2 = 0

	Y_predicted = predict_y(X, params)						# Compute Y-hat

	cost = (1/(2*X.shape[1]))*(np.sum(np.square(Y - Y_predicted)))			# Calculate cost

	db_0 = (2/X.shape[1])*(np.sum(-(Y - Y_predicted))) 				# Calculate gradients 
	db_1 = (2/X.shape[1])*(np.sum(-(Y - Y_predicted)*X[1,:]))
	db_2 = (2/X.shape[1])*(np.sum(-(Y - Y_predicted)*X[2,:]))

	grads = {'db_0': db_0,
			'db_1': db_1,
			'db_2': db_2}							# dictionary to store gradient values

	return grads, cost

# Plotting
def plotting(output):
	'''
	Plotting function

	Arguments:
	output: dictinary of model results, keys output = {'costs': costs,
				'Y_predicted': Y_predicted,
				'Y_true': Y,
				'params': params,
				'learning_rate': learning_rate,
				'num_iteration': num_iteration}
	'''
	plt.figure(1)
	plt.subplot(211)
	x = output['Y_true']
	y = output['Y_predicted']
	#fig = figure()
	plt.plot(x, y, 'bs')

	plt.subplot(212)
	x = np.arange(1, 9000, 100)
	y = output['costs']
	plt.plot(x, y, 'bs')
	plt.show()

	return None

# Optimize
def optimize(X, Y, params, learning_rate=0.01, num_iteration=500, print_cost=True):
	'''
	Implements optimization of the parameters using gradient descent

	Arguments:
	X -- array of feature values 
	Y -- array of label values
	params -- dictionary of paramters
	learning rate -- hyperparameter that controls the rate of descent in gradient descent process, default to 0.01
	num_iteration -- hyperparameter that sets the maximum number of iteration for model to learn optimal paramters, default to 500
	print_cost -- boolean to control the print cost statement, default True

	Returns:
	params -- dictionary of beta parameters
	grads -- dictionary of gradients
	costs -- list of cost per specified time steps
	'''

	beta_0 = params['beta_0']							# Retrieve beta values from dictionary
	beta_1 = params['beta_1']
	beta_2 = params['beta_2']	

	costs = []

	for i in range(num_iteration):

		grads, cost = compute_cost(X, Y, params)				# Get gradients and cost

		db_0 = grads['db_0']							# Retrieve derivatives from grads
		db_1 = grads['db_1']
		db_2 = grads['db_2']

		beta_0 = beta_0 - learning_rate*db_0					# Update rule for paramters 
		beta_1 = beta_1 - learning_rate*db_1
		beta_2 = beta_2 - learning_rate*db_2

		params = {'beta_0': beta_0,
				'beta_1': beta_1,
				'beta_2': beta_2}					# dictionary to store parameter values

		if i % 100 == 0:							# Record cost to costs every 100 iteration
			costs.append(cost)											

		if print_cost and i % 100 == 0:									
			print('The cost at {} iteration is {}.'.format(i, cost) )	# Print cost

	params = {'beta_0': beta_0,
				'beta_1': beta_1,
				'beta_2': beta_2}					# dictionary to store parameter values

	grads = {'db_0': db_0,
			'db_1': db_1,
			'db_2': db_2}							# dictionary to store gradient values

	return params, grads, costs

# linear regression model
def model_linear(X, Y, learning_rate=0.01, num_iteration=500, print_cost=True):
	'''
	Implements linear regression model

	Arguments:
	X -- array of feature values 
	Y -- array of label values
	params -- dictionary of paramters
	learning rate -- hyperparameter that controls the rate of descent in gradient descent process, default to 0.01
	num_iteration -- hyperparameter that sets the maximum number of iteration for model to learn optimal paramters, default to 500
	print_cost -- boolean to control the print cost statement, default True

	Returns:
	output -- linear regression model
	'''

	# initialize parameters
	params =  initialize_parameters()

	# Optimize
	params, grads, costs = optimize(X, Y, params, learning_rate, num_iteration, print_cost)

	# Predict
	Y_predicted = predict_y(X, params)

	# Print test/train errors
	print('Accuracy: {}%'.format((np.sum(Y_predicted == Y)/Y.shape[0])*100))

	# Output dictionary
	output = {'costs': costs,
				'Y_predicted': Y_predicted,
				'Y_true': Y,
				'params': params,
				'learning_rate': learning_rate,
				'num_iteration': num_iteration}

	return output

def main(argv):

	X, Y = get_file(argv)					# Get data from input file
	X_scaled = normalize(X)					# Data normalization
	X_scaled = (np.array(X_scaled)).T 			# Convert to numpy array and transpose
	Y = np.array(Y)						# Conver to numpy array 

	output = model_linear(X_scaled, Y, learning_rate=0.0005, num_iteration=9000, print_cost=True)

	print(output)

if __name__=='__main__':
	main(sys.argv)

