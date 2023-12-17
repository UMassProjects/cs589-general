import numpy as np
from jax import grad, random
import jax.numpy as jnp
import matplotlib.pyplot as plt

datapath = "./"

#################### Task 1 ###################

"""
Implementing the linear classification with Softmax cost;
verify the implementation is correct by achiving zero misclassification.
"""

def run_task1():
	# load in data
	csvname = datapath + '2d_classification_data_v1.csv'
	data = np.loadtxt(csvname,delimiter = ',')


	# take input/output pairs from data
	x = data[:-1, :]
	y = data[-1:, :]
	x_train, y_train = jnp.vstack([jnp.array(x), jnp.ones(jnp.shape(x))]), jnp.array(y)
	def costfn(w):
		return jnp.average(jnp.log(1 + jnp.exp(-y_train * jnp.dot(x_train.T, w))))

	def training_loop(w, cost_fn, max_it, alpha, diminishing_steplength = False):
		grad_cost = grad(cost_fn)
		weight_history, cost_history= [], []
		for i in range(1, max_it + 1):
			if diminishing_steplength:
				alpha = 1/i
			w = w - alpha * grad_cost(w)
			weight_history.append(w)
			cost_history.append(cost_fn(w))
		return weight_history, cost_history

	def get_misclass(w):
		return jnp.sum(jnp.sign(jnp.tanh(jnp.dot(x_train.T, w))) != y_train)

	def get_acc(w):
		return (11 - get_misclass(w))/11

	print(np.shape(x)) # (1, 11)
	print(np.shape(y)) # (1, 11)
	weight, cost = training_loop(np.array([3.0, 3.0]), costfn, 2000, 1, False)
	print(weight[-1])
	print(f"Total number of misclassifications in task 1: {get_misclass(weight[-1])}")
	print(f"Accuracy of model on task 1: {get_acc(weight[-1])} ")
	x_plot = jnp.arange(jnp.min(x), jnp.max(x), 0.01)
	y_plot = jnp.tanh(x_plot * weight[-1][0] + weight[-1][1])

	# plotting cost history
	plt.figure()
	plt.plot(np.arange(0, 2000), cost, 'b')
	plt.xlabel('Iteration')
	plt.ylabel('Cost')
	plt.title('Cost Through Time for Softmax Cost Function')
	plt.savefig('task1_cost.png')

	## PLOTTING THE FIGURE
	plt.figure()
	plt.plot(x_train[:1, :], y_train, 'bo', label = 'data')
	plt.plot(x_plot, y_plot, 'r', label = 'tanh curve')
	plt.axhline(y=0, color='k', lw=0.3)
	plt.axvline(x=0, color='k', lw=0.3)
	plt.xlabel('x')
	plt.ylabel('Prediction')
	plt.title('Linear Classification with Softmax Cost and Tanh Function')
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())
	# plt.show()
	plt.savefig('task1.png')




#################### Task 2 ###################

"""
Compare the efficacy of the Softmax and
the Perceptron cost functions in terms of the
minimal number of misclassifications each can
achieve by proper minimization via gradient descent
on a breast cancer dataset.
"""

def run_task2():
	# data input
	csvname = datapath + 'breast_cancer_data.csv'
	data = np.loadtxt(csvname,delimiter = ',')
	# get input and output of dataset
	x = data[:-1, :]
	y = data[-1:, :]
	# print(jnp.shape(jnp.array(x)))
	x_train, y_train = jnp.vstack([jnp.array(x), jnp.ones((1,699))]), jnp.array(y)

	def softmax(w):
		return jnp.average(jnp.log(1 + jnp.exp(-y_train * jnp.dot(x_train.T, w))))
	def perceptron(w):
		return jnp.average(jnp.maximum(0, -y_train * jnp.dot(x_train.T, w)))
	def training_loop(w, alpha, max_it, cost_fn, diminishing_steplength = False):
		grad_cost = grad(cost_fn)
		weight_history, cost_history, acc_history= [], [], []
		for i in range(1, max_it + 1):
			if diminishing_steplength:
				alpha = 1/i
			w = w - alpha * grad_cost(w)
			weight_history.append(w)
			cost_history.append(cost_fn(w))
			acc_history.append(get_acc(w))
		return weight_history, cost_history, acc_history
	def get_misclass(w):
		return jnp.sum(jnp.sign(jnp.dot(x_train.T, w) * y_train) <= 0)
	def get_acc(w):
		return (699 - get_misclass(w)) / 699
	key = random.PRNGKey(40)
	default_iteration = 2000
	#retrieving the weight history
	wsoftmax, csoftmax, asoftmax = training_loop(random.normal(key, (9,)), 0.027, default_iteration, softmax, False)
	wperceptron, cperceptron, aperceptron = training_loop(random.normal(key, (9,)), 0.027, default_iteration, perceptron, False)

	# retrieving the final accuracy
	print("Accuracy of Softmax is: ", asoftmax[-1])
	print("Accuracy of Perceptron is: ", aperceptron[-1])
	print("Misclassification of Softmax is: ", get_misclass(wsoftmax[-1]))
	print("Misclassification of Perceptron is: ", get_misclass(wperceptron[-1]))


	# plotting the figure
	plt.figure()
	plt.plot(np.arange(0, default_iteration), aperceptron, 'b', label = 'Perceptron')
	plt.plot(np.arange(0, default_iteration), asoftmax, 'r', label = 'Softmax')
	plt.xlabel('Iteration')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Through Time for Each of the Cost Functions')
	plt.legend()
	plt.savefig('task2_accuracy.png')


	# plotting the cost through time
	plt.figure()
	plt.plot(np.arange(0, default_iteration), csoftmax, 'r', label = 'Softmax')
	plt.plot(np.arange(0, default_iteration), cperceptron, 'b', label = 'Perceptron')
	plt.xlabel('Iteration')
	plt.ylabel('Cost')
	plt.title('Cost Through Time for Each of the Cost Functions')
	plt.legend()
	plt.savefig('task2_cost.png')


	print(np.shape(x)) # (8, 699)
	print(np.shape(y)) # (1, 699)


if __name__ == '__main__':
	run_task1()
	run_task2()



