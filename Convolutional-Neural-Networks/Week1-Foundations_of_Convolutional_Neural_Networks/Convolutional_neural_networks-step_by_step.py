'''
Convolutional Neural Networks: Step by Step
Coursera: Concolutional Neural Network
Week 1 project: CNN Step by Step
				Implementation of convolution (CONV) and pooling (POOL)
				including forward propagation and (optionally) backward propagation
'''

import numpy as np 
import h5py
import matplotlib.pyplot as plt 

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # est default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2

np.random.seed(1)

# zero pad
def zero_pad(X, pad):
	'''
	Pad with zerp all images of the dataset X. The padding is applied to the 
	height and width of an image

	Argument:
	X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
	pad -- integer, amount of padding around each image on vertical and horizontal dimensions

	Returns:
	x_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
	'''

	X_pad = np.pad(X, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode='constant', constant_values=(0))

	return X_pad

def conv_single_step(a_slice_prev, W, b):
	'''
	Apply one filter defined by paramters W on a single slice (a_slice_prev) of the 
	output activation of the previous layer

	Arguments:
	a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
	W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
	b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

	Returns:
	Z -- a scalar value, result of convolving the sliding window (W, b) on a 
	slice x of the input data
	'''

	# Element-wise product between a_slice_prev and W. Do not add the bias yet
	# print(a_slice_prev.shape, W.shape)
	S = np.multiply(a_slice_prev, W)

	# Sum over all entries of te volume s
	Z = np.sum(S)

	# Add bias b to Z. Cast b to a float() so that X results in a scalar value
	Z = np.float(Z + b)

	return Z

def conv_forward(A_prev, W, b, hparameters):
	'''
	Implements the forward propagation for a convolution function

	Arguments:
	A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
	b -- Biases, numpy array of shape (1, 1, 1, n_C)
	hparameters -- python dictionary containing 'stride and 'pad'
	
	Returns:
	Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
	cache -- cache of values needed for the conv_backward() function
	'''

	# Retrieve dimensions from A_prev's shape 
	(m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
	# print('A_prev shape: ', np.shape(A_prev))

	# Retrieve dimensions from W's shape
	(f, f, n_C_prev, n_C) = np.shape(W)
	# print('W shape: ', np.shape(W) )

	# Retrieve information from 'hparameters'
	stride = hparameters.get('stride', '')
	pad = hparameters.get('pad', '')

	# Computer the dimension of the CONV ouput volume using the formula
	# Use int() for floor
	n_H = int((n_H_prev + 2*pad - f)/stride) + 1
	n_W = int((n_W_prev + 2*pad - f)/stride) + 1

	# Initialize the output volume Z with zeros.
	Z = np.zeros((m, n_H, n_W, n_C))
	# print('Z output: ', np.shape(Z))

	# Create A_prev_pad by padding A_prev
	A_prev_pad = zero_pad(A_prev, pad)
	# print("input shape before padding: ", np.shape(A_prev))
	# print("input shape after padding: ", np.shape(A_prev_pad))
	# print('looping over m = {}, n_H = {}, n_W = {}, n_C = {}'.format( m, n_H, n_W, n_C))
	# print ('f: ',f)

	for i in range(m):					# loop over the batch of training examples
		a_prev_pad = A_prev_pad[i]		# select ith training example's padded activation 
		# print('select ith example padded activation: ', np.shape(a_prev_pad))
		for h in range(n_H):			# loop over the vertical axis of the output volume
			for w in range(n_W):		# loop over the horizontal axis of the output volume
				for c in range(n_C):	# loop over channels (= #filters) of the output volume
					
					# find the corners of the current 'slice'
					#print(c)
					#print('h: ', h, h+stride)
					vert_start = h * stride
					vert_end = vert_start + f
					horiz_start = w * stride
					horiz_end = horiz_start + f

					# Use the corners to define the (3D) slice of a_prev_pad 
					a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
					#print('a slice ', np.shape(a_slice_prev))
					# p=W[:,:,:,i]
					# print(np.shape(p))
					# print(p)
					# pp=b[:,:,:,i]
					# print(np.shape(pp))
					# print(pp)
					
					# Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
					Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])

	# Making sure your output shape is correct
	assert(Z.shape == (m, n_H, n_W, n_C))

	# Save information in 'cache' for the backprop
	cache = (A_prev, W, b, hparameters)

	return Z, cache
	
def pool_forward(A_prev, hparameters, mode = 'max'):
	'''
	Implements the forward pass of the pooling layer

	Arguments:
	A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	hparameters -- python dictionary containing 'f' and 'stride'
	mode -- the pooling mode you would like to use, defined as a string ('max' or 'average')

	Returns:
	A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
	cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
	'''

	# Retrieve dimensions from the input shape
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	print ('shape of input: ',A_prev.shape)

	# Retrieve hyperparameters from 'hparameters'
	f = hparameters['f']
	stride = hparameters['stride']
	print("f = {}, stride = {}".format(f,stride))

	# Define the dimensions of the output
	n_H = int(1 + (n_H_prev - f)/stride) 
	n_W = int(1 + (n_W_prev - f)/stride)
	n_C = n_C_prev

	# initialize output matrix A
	A = np.zeros((m, n_H, n_W, n_C))
	print('shape of A: ', A.shape)

	### START CODE HERE ###
	for i in range(m):					# loop over the training examples
		for h in range(n_H):			# loop over the vertical axis of the output volume
			for w in range(n_W):		# loop over the horizontal axis of the output volume
				for c in range(n_C):	# lop over the channels of the output volume

					# Find the corners of the current 'slice' (~4 lines)
					vert_start = h * stride
					vert_end = vert_start + f
					horiz_start = w * stride
					horiz_end = horiz_start + f
					#print('vert_start = {}, vert_end = {}, horiz_start = {}, horiz_end = {}'.format(vert_start, vert_end, horiz_start, horiz_end))

					# Use the corners to define the current slice on the ith traning example of A_prev, channel c. (~1 line)
					a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
					print(a_prev_slice.shape)

					# Computer the pooling operation on the slice. 
					# Use an if statement to differentiate the modes.
					# Use np.max/np.mean
					if mode == "max":
						A[i, h, w, c] = np.max(a_prev_slice)
					elif mode == 'average':
						A[i, h, w, c] = np.mean(a_prev_slice)

	### END CODE HERE ###

	# Store the input and hparameters in 'cache' for pool_backward()
	cache = (A_prev, hparameters)

	# Making sure your output shape is correct
	assert(A.shape == (m, n_H, n_W, n_C))

	return A, cache

#np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)