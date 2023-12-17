import jax.numpy as jnp
import numpy as np
from jax import grad, random
import matplotlib.pyplot as plt
datapath = "./"

#################### Task 3 ###################

"""
Implementing the multi-class classification with Softmax cost;
verify the implementation is correct by achiving small misclassification rate.
"""


# A helper function to plot the original data
def show_dataset(x, y):
  y = y.flatten()
  num_classes = np.size(np.unique(y.flatten()))
  accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
  # initialize figure
  plt.figure()

  # color current class
  for a in range(0, num_classes):
    t = np.argwhere(y == a)
    t = t[:, 0]
    plt.scatter(
      x[0, t],
      x[1, t],
      s=50,
      color=accessible_color_cycle[a],
      edgecolor='k',
      linewidth=1.5,
      label="class:" + str(a))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(bbox_to_anchor=(1.1, 1.05))

  plt.savefig("data.png")
  plt.close()

def show_dataset_labels(x, y, modelf, n_axis_pts=120):
  y = y.flatten()
  num_classes = np.size(np.unique(y.flatten()))
  accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
  # initialize figure
  plt.figure()

  # fill in label regions using scatter points
  # get (x1, x2) for plot region
  anyax = np.linspace(0.05, 0.95, num=n_axis_pts)
  xx = np.meshgrid(anyax, anyax)
  xx_vars = np.reshape(xx, (2, n_axis_pts **2))
  # get class weights from classifier model
  z = modelf(xx_vars)
  # get class label from model output
  y_hat = z.argmax(axis=1)

  for a in range(0, num_classes):
    t = np.argwhere(y_hat == a)
    t = t[:, 0]
    plt.scatter(
      xx_vars[0, t],
      xx_vars[1, t],
      s=5,
      color=accessible_color_cycle[a],
      linewidth=1.5,
      label="class:" + str(a))

  # color current class
  for a in range(0, num_classes):
    t = np.argwhere(y == a)
    t = t[:, 0]
    plt.scatter(
      x[0, t],
      x[1, t],
      s=50,
      color=accessible_color_cycle[a],
      edgecolor='k',
      linewidth=1.5,
      label="class:" + str(a))
    plt.xlabel("x1")
    plt.ylabel("x2")
  plt.legend(bbox_to_anchor=(1.1, 1.05))
  plt.savefig("classifier_label_regions.png")
  plt.close()


def run_task3():
  # load in dataset
  data = np.loadtxt(datapath + '4class_data.csv', delimiter=',')

  # get input/output pairs
  x = data[:-1, :]
  y = data[-1:, :]
  print(np.shape(x))
  print(np.shape(y))

  show_dataset(x, y)
  def model(x, w):
    a = w[0] + jnp.dot(x.T, w[1:])
    return a.T

  alpha = 1.1e-1
  lam = 1e-5
  def multiclass_softmax(w):
    all_evals= model(x, w)
    a = jnp.log(jnp.sum(jnp.exp(all_evals), axis=0))
    b = all_evals[y.astype(int).flatten(),jnp.arange(np.size(y))]
    cost = jnp.sum(a - b)
    cost = cost + lam * jnp.linalg.norm(w[1:], 'fro')**2
    return cost/float(np.size(y))
  def get_accuracy(w):
    all_evals= model(x, w)
    y_hat = all_evals.argmax(axis=0)
    return np.sum(y_hat == y.astype(int).flatten())/float(np.size(y))
  def training_loop(w, cost_fn, max_it, alpha, diminishing_steplength = False):
    grad_cost = grad(cost_fn)
    # weight_history = []
    for i in range(1, max_it + 1):
      if diminishing_steplength:
        alpha = 1/i
      w = w - alpha * grad_cost(w)
      # weight_history.append(w)
    # return weight_history
    return w
  def train(xs):
    weights = training_loop(np.random.randn(3, 4), multiclass_softmax, 5000, alpha, False)
    # w = weights[-1]
    w = weights
    print(w, np.shape(xs), np.shape(model(xs, w)))
    print(f"The final accuracy is: {get_accuracy(w)}")
    print(f"The final cost is: {multiclass_softmax(w)}")
    return model(xs, w).T

  show_dataset_labels(x,y, train)




  # TODO: fill in your code


if __name__ == '__main__':
  run_task3()