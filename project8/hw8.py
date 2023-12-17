import numpy as np
import pandas as pd
from jax import grad
import jax.numpy as jnp
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt

datapath = "./"


def train_test_split(*arrays, test_size=0.2, shuffle=True, rand_seed=1):
    # set the random state if provided
    np.random.seed(rand_seed)

    # initialize the split index
    array_len = len(arrays[0].T)
    split_idx = int(array_len * (1 - test_size))

    # initialize indices to the default order
    indices = np.arange(array_len)

    # shuffle the arrays if shuffle is True
    if shuffle:
        np.random.shuffle(indices)

    # Split the arrays
    result = []
    for array in arrays:
        if shuffle:
            array = array[:, indices]
        train = array[:, :split_idx]
        test = array[:, split_idx:]
        result.extend([train, test])

    return result

def split_k_fold(array, k, j, round_down = False):
    """
    array = train set split
    k = number of folds
    j = current fold
    round_down = whether to discard the last few entries
    """
    entries = len(array.T)
    fold_size = entries//k
    res = []
    for i in range(k):
        res.append(array[:, i*fold_size:(i+1)*fold_size])
    if not round_down:
        for i in range(entries % fold_size):
            sup = np.array(res[i])
            res[i] = np.concatenate((sup, array[:, -(i+1):-i]), axis = 1)
    return np.concatenate(res[:j] + res[j+1:], axis = 1), res[j]

def model(x, w):
	"""
	input:
	- x: shape (N, P)
	- W: shape (N+1, C)

	output:
	- prediction: shape (C, P)
	"""
	f = x
	o = jnp.ones((1, np.shape(f)[1]))
	f = jnp.vstack((o,f))
	a = jnp.dot(f.T,w)
	return a.T

def softmax(w, x_p, y_p, lam):
    all_evals = model(x_p,w)
    return np.reshape(jnp.average(jnp.log(1 + jnp.exp(-y_p*all_evals)), axis = 1) + lam*jnp.sum(jnp.abs(w)), ())

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

def training_loop(x, y, w, lam, cost_fn, max_it, alpha):
    grad_cost = grad(cost_fn)
    c_history = [cost_fn(w, x, y, lam)]
	# acc_history = [accuracy(y, jnp.argmax(model(x, w), axis=0))]
	# for i in range(max_it):
    #     if i % 1000 == 0:
    #         print(f"iteration {i}")
	# 	w = w - alpha * grad_cost(w, x, y, lam)
	# 	# w_history.append(w)
	# 	# acc_history.append(accuracy(y, jnp.argmax(model(x, w), axis=0)))
    for i in range(max_it):
        # When cost stops decreasing, decrease learning rate by 0.1
        if i % 200 == 0 and i > 0 and np.mean(c_history[-100:-50]) - np.mean(c_history[-50:]) < 1e-8:
            if alpha > 1e-4:
                alpha = alpha * 0.1
            # early stop as we have achieved the lowest alpha
            else:
                print(f"early stop at k = {i}")
                break

        w = w - alpha * grad_cost(w, x, y, lam)
        c_history.append(cost_fn(w, x, y, lam))
    return w, c_history

def accuracy(y, y_pred):
    return jnp.sum(jnp.sign(y) == y_pred)/len(y)

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

def run_task1():
    csvname = datapath + 'new_gene_data.csv'
    data = np.loadtxt(csvname, delimiter=',')
    x = data[:-1, :]
    y = data[-1:, :]


    print(np.shape(x))  # (7128, 72)
    print(np.shape(y))  # (1, 72)

    x = pca_sphere(x)

    np.random.seed(0)  # fix randomness
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, rand_seed=0)

    # Preprocessing
    # TODO: Apply normalization method or other preprocessing if needed

    # K-fold Cross Validation
    k = 5  # Number of folds
    best_accuracy = 0
    best_hyperparameters = None
    lr = [10**(-i) for i in range(3)]
    lamda = [10**(-j) for j in range(1, 4)]
    hyperparameter_grid = [{'lr': i, 'lam': j} for i in lr for j in lamda]

    for hyperparameters in hyperparameter_grid:
        print(f"Begin training hyperparameters: {hyperparameters}")
        alpha = hyperparameters['lr']
        lam = hyperparameters['lam']
        fold_accuracies = []
        for fold in range(k):
            w = np.random.randn(x_train.shape[0] + 1)  # Initialize model parameters

            # Split data into train and validation sets
            x_train_fold, x_val_fold = split_k_fold(x_train, k, fold)
            y_train_fold, y_val_fold = split_k_fold(y_train, k, fold)
            # Train model with current hyperparameters
            # training_loop(x, y, w, lam, cost_fn, max_it, alpha)
            weight, c_history = training_loop(x_train_fold, y_train_fold, w, lam, softmax, 1000, alpha)

            # Evaluate model on validation set
            acc = accuracy(jnp.tanh(model(x_val_fold, weight)), y_val_fold)
            fold_accuracies.append(acc)
            print(f"progress: {fold + 1}/{k}", end="\r")

        # Calculate average validation accuracy for current hyperparameters
        average_accuracy = np.mean(fold_accuracies)
        print(average_accuracy)
        # Update best hyperparameters if current accuracy is higher
        if average_accuracy >= best_accuracy:
            best_accuracy = average_accuracy
            best_hyperparameters = hyperparameters

    # Train model with best hyperparameters on entire training set

    print(f'Best hyperparameters were:{best_hyperparameters}, whose average fold accuracy was: {best_accuracy}')
    w = np.random.randn(x_train.shape[0] + 1)  # Initialize model parameters
    weight_final, c_history = training_loop(x_train, y_train, w, best_hyperparameters['lam'], softmax, 2000, best_hyperparameters['lr'])
    # weight_final, c_history = training_loop(x_train, y_train, w, 0.1, softmax, 2000, 0.1)


    # Evaluate model on test set

    acc_final = accuracy(jnp.tanh(model(x_test, weight_final)), y_test)
    print(f'Test accuracy: {acc_final}')
    print(f'five most influential genes {np.abs(weight_final[1:]).argsort()[::-1][:5]}')

    # Plot cost vs. iteration
    plt.plot(c_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iteration')
    plt.savefig("cost_vs_iteration2.png")


if __name__ == '__main__':
    run_task1()
