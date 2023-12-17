# hw3.py
import jax.numpy as jnp
from jax import grad
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
import numpy as np
import matplotlib.pyplot as plt


datapath="./"

#################### Task 1 ###################

"""
Fit a linear regression model to the student debt data
All parts marked "TO DO" are for you to construct.
"""


def run_task1():
	# import the dataset
	csvname = datapath + 'student_debt_data.csv'
	data = np.loadtxt(csvname,delimiter=',')

	# extract input - for this dataset, these are times
	x = data[:,0]
	# extract output - for this dataset, these are total student debt
	y = data[:,1]

	print(np.shape(x))
	print(np.shape(y))

	x_train, y_train = jnp.array(x), jnp.array(y)
	def model(w):
		return w[0] + jnp.sum(w[1:] * x_train[:, jnp.newaxis], axis=1)
	def loss_fn(w):
		return jnp.mean((model(w) - y_train)**2)
	def gradient_descent(w, max_it, alpha, diminishing_steplength = False):
		gradient = grad(loss_fn)
		weight_history = [w]
		loss_history = [loss_fn(w)]
		for i in range(1, max_it + 1):
			alpha = 1/i if diminishing_steplength else alpha
			w = w - alpha * gradient(w)
			weight_history.append(w)
			loss_history.append(loss_fn(w))
		return weight_history, loss_history


	# TODO: fit a linear regression model to the data
	weight_history, loss_history = gradient_descent(jnp.array([-169.0, 10472.0]), 150, 1.75e-8, False)
	w = weight_history[-1]
	print("Task 01, equation of the fitted line is", w)
	print("Predicted Student Debt in 2030 is", w[0] + w[1] *2030)
	plt.figure()
	plt.plot(x, y, 'o')
	plt.plot(x, model(weight_history[-1]), 'r')
	plt.xlabel('Year')
	plt.ylabel('Debt')
	plt.title('Debt Regression')
	# plt.show()
	plt.savefig('task1.png')

	plt.figure()
	plt.plot(loss_history, 'o')
	plt.savefig('task1_loss.png')



#################### Task 2 ###################

"""
Compare the least squares and the least absolute deviation costs
All parts marked "TO DO" are for you to construct.
"""

def run_task2():
	# load in dataset
	data = np.loadtxt(datapath + 'regression_outliers.csv',delimiter = ',')
	x = data[:-1,:].flatten()
	y = data[-1:,:].flatten()
	x_train, y_train = jnp.array(x), jnp.array(y)
	print(np.shape(x_train))
	print(np.shape(y_train))
	# print(np.shape(jnp.array([0.1, 0.1])))

	# TODO: fit two linear models to the data
	def model(w):
		return w[0] + jnp.sum(w[1:] * x_train[:, jnp.newaxis], axis=1)
	def loss_fn1(w):
		return jnp.mean((model(w) - y_train)**2)
	def loss_fn2(w):
		return jnp.mean(jnp.abs(model(w) - y))
	def gradient_descent(w, loss_fn, max_it, alpha, diminishing_steplength = False):
		gradient = grad(loss_fn)
		weight_history = [w]
		loss_history = [loss_fn(w)]
		for i in range(1, max_it + 1):
			alpha = 1/i if diminishing_steplength else alpha
			w = w - alpha * gradient(w)
			weight_history.append(w)
			loss_history.append(loss_fn(w))
		return weight_history, loss_history
	weight_history1, lost_history1 = gradient_descent(jnp.array([0.1, 0.1]), loss_fn1, 1000, 1e-2, False)
	weight_history2, lost_history2 = gradient_descent(jnp.array([0.1, 0.1]), loss_fn2, 1000, 1e-2, False)
	print(weight_history1[-1], weight_history2[-1])
	plt.figure()
	plt.plot(x, y, 'o')
	plt.plot(x, model(weight_history1[-1]), 'r')
	plt.plot(x, model(weight_history2[-1]), 'b')
	plt.title('Least absolute deviation vs Least square in dataset with outliers')
	plt.legend(['Least absolute deviation', 'Least square'])
	# plt.show()
	plt.savefig('task2.png')


if __name__ == '__main__':
	run_task1()
	run_task2()


