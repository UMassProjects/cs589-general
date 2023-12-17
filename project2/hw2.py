

import jax.numpy as jnp
import numpy as np
from jax import grad
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html


import matplotlib.pyplot as plt

#################### Task 1 ###################

"""
In this exercise you will implement gradient descent using the hand-computed derivative.
All parts marked "TO DO" are for you to construct.
"""
def cost_func(w):
	"""
	Params:
	- w (weight)

	Returns:
	- cost (value of cost function)
	"""
	return 0.02 * (w**4 + w**2 + 10*w)

def gradient_func(w):
	"""
	Params:
	- w (weight)

	Returns:
	- grad (gradient of the cost function)
	"""
	return 0.02*(4*w**3 + 2*w + 10)

def gradient_descent(g, gradient, alpha,max_its,w):
	"""
	Params:
	- g (input function),
	- gradient (gradient function that computes the gradients of the variable)
	- alpha (steplength parameter),
	- max_its (maximum number of iterations),
	- w (initialization)

	Returns:
	- cost_history
	"""

	# run the gradient descent loop
	cost_history = [g(w)]        # container for corresponding cost function history
	for k in range(1,max_its+1):
		# TODO: evaluate the gradient, store current weights and cost function value
		grad = gradient(w)
		w -= alpha*grad
		# collect final weights
		cost_history.append(g(w))
	return cost_history



def run_task1():
	print("run task 1 ...")
	# TODO: Three seperate runs using different steplength
	cost1 = gradient_descent(cost_func, gradient_func, 1, 1000, 2)
	cost2 = gradient_descent(cost_func, gradient_func, 0.1, 1000, 2)
	cost3 = gradient_descent(cost_func, gradient_func, 0.01, 1000, 2)

	# create a plot to plot the 3 costs in one figure, with different colors
	plt.figure()
	plt.title('Task 1 Plot With Different Alpha')
	plt.plot(cost1, label='alpha = 1')
	plt.plot(cost2, label='alpha = 0.1')
	plt.plot(cost3, label='alpha = 0.01')
	plt.legend()
	plt.savefig('task1.jpg')


	print("task 1 finished")



#################### Task 2 ###################

"""
In this exercise you will implement gradient descent
using the automatically computed derivative.
All parts marked "TO DO" are for you to construct.
"""
def absolute(w):
	return abs(w)


def gradient_descent_auto(g,alpha,max_its,w, diminishing_alpha=False):
	"""

	gradient descent function using automatic differentiator
	Params:
	- g (input function),
	- alpha (steplength parameter),
	- max_its (maximum number of iterations),
	- w (initialization)

	Returns:
	- weight_history
	- cost_history

	"""
	# TODO: compute gradient module using jax


	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w)]          # container for corresponding cost function history
	for k in range(1, max_its+1):
		# TODO: evaluate the gradient, store current weights and cost function value
		gradient = grad(g)(w)
		if diminishing_alpha:
			alpha = alpha / k
		w -= alpha*gradient
		# record weight and cost
		weight_history.append(w)
		cost_history.append(g(w))
	return weight_history,cost_history

def run_task2():
	print("run task 2 ...")
	# TODO: implement task 2
	weight4, cost4 = gradient_descent_auto(absolute, 0.5, 20,2.0, diminishing_alpha=False)
	weight5, cost5 = gradient_descent_auto(absolute, 1,20,2.0, diminishing_alpha=True)

	# create a plot to plot the 2 costs in one figure, with different colors
	# create a new plot
	plt.figure()
	plt.title('Task 2 Plot With Different Alpha')
	plt.plot(cost4, label='alpha = 0.5')
	plt.plot(cost5, label='alpha diminishes')
	plt.legend()
	plt.savefig('task2.jpg')
	print("task 2 finished")

if __name__ == '__main__':
	run_task1()
	run_task2()



