# hw7.py
import jax.numpy as jnp 
from jax import grad, random 
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt 
datapath = "./"


#################### Task 1 ###################

###### Helper functions for task 1 ########
# multi-class linear classification model 
def model(x, w): 
	"""
	input: 
	- x: shape (N, P)  
	- W: shape (N+1, C) 

	output: 
	- prediction: shape (C, P) 
	"""
	# option 1: stack 1 
	f = x   
	# print("before stack 1, x.shape: ", f.shape)

	# tack a 1 onto the top of each input point all at once
	o = jnp.ones((1, np.shape(f)[1]))
	f = jnp.vstack((o,f))

	# print("after stack 1, the X.shape:", f.shape)

	# compute linear combination and return
	a = jnp.dot(f.T,w)

	# option 2: 
	# a = w[0, :] + jnp.dot(x.T, w[1:, :])
	return a.T


# multi-class softmax cost function 
def multiclass_softmax(w, x_p, y_p):     
	"""
	Args:
	 	- w: parameters. shape (N+1, C), C= the number of classes
	 	- x_p: input. shape (N, P) 
		- y_p: label. shape (1, P)
	Return: 
		- softmax cost: shape (1,)
	"""
    
	# pre-compute predictions on all points
	all_evals = model(x_p,w)
	# print(f"all_evals[:, 0:5].T={all_evals[:, 0:5].T}")

	# logsumexp trick
	maxes = jnp.max(all_evals, axis=0)
	a = maxes + jnp.log(jnp.sum(jnp.exp(all_evals - maxes), axis=0))

	# compute cost in compact form using numpy broadcasting
	b = all_evals[y_p.astype(int).flatten(), jnp.arange(np.size(y_p))]
	cost = jnp.sum(a - b)

	# return average
	return cost/float(np.size(y_p))

def training_loop(x, y, w, cost_fn, max_it, alpha):
	grad_cost = grad(cost_fn)
	w_history = [w]
	c_history = [cost_fn(w, x, y)]
	acc_history = [accuracy(y, jnp.argmax(model(x, w), axis=0))]
	for i in range(max_it):
		w = w - alpha * grad_cost(w, x, y)
		w_history.append(w)
		c_history.append(cost_fn(w, x, y))
		acc_history.append(accuracy(y, jnp.argmax(model(x, w), axis=0)))
	return w_history, c_history, acc_history


def standard_normalize(x):
	"""
	Args:
		- x: input data. shape (N, P)
	Return:
		- normalized data: shape (N, P)
	"""
	x_mean = np.mean(x, axis=1)[:, np.newaxis]
	x_std = np.std(x, axis=1)[:, np.newaxis]

	# exploding exponential checks

	idx = np.argwhere(x_std < 1e-2) 
	if len(idx) > 0:
		ind = [v[0] for v in idx]
		adjust = np.zeros((x_std.shape))
		adjust[ind] = 1.0
		x_std += adjust

	return (x - x_mean) / x_std

def accuracy(y, y_pred):
	return jnp.mean(y == y_pred)

def compute_pcs(X, lam):
		P = float(X.shape[1])
		Cov = 1/P * jnp.dot(X, X.T) + lam*jnp.eye(X.shape[0])

		D, V = jnp.linalg.eigh(Cov)
		return D, V


def pca_sphere(x):
	"""
	Args:
		- x: input data. shape (N, P)
	Return:
		- normalized data: shape (N, P)
	"""
	# compute eigenvalues and eigenvectors of covariance matrix
	D, V = compute_pcs(x, 1e-5)
	D = jnp.abs(D)
	# compute the projection matrix
	P = jnp.dot(V.T, x)
	return P/(D[:,np.newaxis]**0.5)

def plot_cost_task1(arr, name, color):
	# x = np.arange(len(arr))
	plt.figure()
	plt.plot(arr, label=f'MNIST_{name}', color=color)
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title(f'Cost through iterations of MNIST_{name}')
	plt.legend()
	plt.savefig(f'task1_{name}_cost.png')

def run_task1(): 
	print("Task 1 begins")
	# import MNIST
	x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
	# re-shape input/output data
	x = np.array(x.T)
	y = np.array([int(v) for v in y])[np.newaxis,:]
	np.random.seed(42)
	w = np.random.randn(x.shape[0] + 1, 10) # num of input features x number of classes (10 because we are dealing with digits)

	print(jnp.shape(x)) # (784, 70000)
	print(jnp.shape(y)) # (1, 70000)

	x_train = x[:, :50000]
	y_train = y[:, :50000]

	x_train_std = standard_normalize(x_train)
	x_train_pca = pca_sphere(x_train_std)
	# outputing the cost and accuracy of the model
	for i in range(-3, 4):
		lam = 10**i
		mnist, mnist_cost, mnist_acc = training_loop(x_train, y_train, w, multiclass_softmax, 10, lam)
		mnist_std, mnist_std_cost, mnist_std_acc = training_loop(x_train_std, y_train, w, multiclass_softmax, 10, lam)
		mnist_pca, mnist_pca_cost, mnist_pca_acc = training_loop(x_train_pca, y_train, w, multiclass_softmax, 10, lam)
		print(f'lambda={lam}, cost={mnist_cost[-1]}, accuracy={mnist_acc[-1]}')
		print(f'lambda={lam}, cost={mnist_std_cost[-1]}, accuracy={mnist_std_acc[-1]}')
		print(f'lambda={lam}, cost={mnist_pca_cost[-1]}, accuracy={mnist_pca_acc[-1]}')

	# x_test, y_test = x[:, 50000:], y[:, 50000:]
	mnist, mnist_cost, mnist_acc = training_loop(x_train, y_train, w, multiclass_softmax, 10, 1e-2)
	mnist_std, mnist_std_cost, mnist_std_acc = training_loop(x_train_std, y_train, w, multiclass_softmax, 10, 10)
	mnist_pca, mnist_pca_cost, mnist_pca_acc = training_loop(x_train_pca, y_train, w, multiclass_softmax, 10, 100)
	
	# print(mnist_cost)
	# print(mnist_std_cost)
	# print(mnist_pca_cost)

	#plotting the performance of the model
	plt.figure()
	plt.plot(mnist_cost, label='MNIST')
	plt.plot(mnist_std_cost, label='MNIST Standardized')
	plt.plot(mnist_pca_cost, label='MNIST PCA')
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title('MNIST Costs')
	plt.legend()
	plt.savefig('mnist_training.png')

	plt.figure()
	plt.plot(mnist_acc, label='MNIST')
	plt.plot(mnist_std_acc, label='MNIST Standardized')
	plt.plot(mnist_pca_acc, label='MNIST PCA')
	plt.xlabel('Iterations')
	plt.ylabel('Accuracy')
	plt.title('MNIST Accuracy')
	plt.legend()
	plt.savefig('mnist_training_acc.png')

	plot_cost_task1(mnist_cost, 'raw', 'red')
	plot_cost_task1(mnist_std_cost, 'standardized', 'blue')
	plot_cost_task1(mnist_pca_cost, 'pca', 'green')

	print("Finished task 1!")
	print("##############################################")

##################

def plot_histogram(arr, lam):

    x = np.arange(1, len(arr))
    plt.figure()
    plt.bar(x, arr[1:])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Histogram of Feature touching weights for lambda={lam}')
    
    # plt.xticks(x)
    # plt.ylim([min(arr)-1, max(arr)+1])
    
    plt.savefig(f'histogram_{lam}.png')

def plot_cost(arr, lam):
	# x = np.arange(len(arr))
	plt.figure()
	plt.plot(arr, label=f'lambda={lam}')
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title(f'Cost history of model with lambda={lam}')
	plt.savefig(f'cost_{lam}.png')
def run_task2(): 
	print("Task 2 begins")
	# load in data
	csvname =  datapath + 'boston_housing.csv'
	data = np.loadtxt(csvname, delimiter = ',')
	x = data[:-1,:]
	y = data[-1:,:] 

	print(np.shape(x))
	print(np.shape(y))
	# input shape: (13, 506)
	# output shape: (1, 506)

	# standardize the input data
	x_std = standard_normalize(x)
	x_pca = pca_sphere(x_std)
	print(np.shape(x_std))
	
	# iters = 100

	def L1(w, lam):
		return lam * jnp.sum(jnp.abs(w[1:]))

	def l1_regularize_cost(w, x_p, y_p, lam):
		cost = jnp.sum((model(x_p, w) - y_p) ** 2)
		cost += L1(w, lam)
		return cost / float(y_p.size)
	def training_loop(x, y, w, lam, cost_fn, max_it, alpha):
		grad_cost = grad(cost_fn)
		w_history = [w]
		c_history = [cost_fn(w, x, y, lam)]
		# acc_history = [accuracy(y, jnp.argmax(model(x, w), axis=0))]
		for i in range(max_it):
			w = w - alpha * grad_cost(w, x, y, lam)
			w_history.append(w)
			c_history.append(cost_fn(w, x, y, lam))
			# acc_history.append(accuracy(y, jnp.argmax(model(x, w), axis=0)))
		return w_history, c_history
	np.random.seed(42)
	# w = np.random.rand(x.shape[0]+1)
	lamda = [0, 50, 100, 150]
	for lam in lamda:
		w = np.random.rand(x.shape[0]+1)
		w_history, c_history = training_loop(x_std, y, w, lam, l1_regularize_cost, 100, 0.03)
		plot_histogram(w_history[-1], lam)
		plot_cost(c_history, lam)
		print(f'lambda={lam} plots completed!, final_cost={c_history[-1]}')

if __name__ == '__main__':
	run_task1()
	run_task2() 




	