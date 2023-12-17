import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)

datapath = "./"

#################### Task 1 ###################

# Implement PCA
# You should be able to implement your own PCA by using numpy only.

def run_task1():
	# load in dataset
	csvname = datapath + '2d_span_data.csv'
	x = np.loadtxt(csvname, delimiter = ',')

	print(np.shape(x)) # (2, 50)
	def center(X):
		X_means = np.mean(X, axis = 1)[:, np.newaxis]
		X_normalized = X - X_means
		return X_normalized

	def compute_pcs(X, lam):
		P = float(X.shape[1])
		Cov = 1/P * np.dot(X, X.T) + lam*np.eye(X.shape[0])

		D, V = np.linalg.eigh(Cov)
		return D, V

	def pca_transform_data(X, **kwargs):
		num_components = X.shape[0]
		if 'num_components' in kwargs:
			num_components = kwargs['num_components']
		lam = 1e-7
		if 'lam' in kwargs:
			lam = kwargs['lam']

		D, V=  compute_pcs(X, lam)
		V = V[:, -num_components:]
		D = D[-num_components:]

		W = np.dot(V.T, X)
		return W, V

	x = center(x)
	W, C = pca_transform_data(x)
	origin = np.array([[0, 0],[0, 0]]) # origin point
	plt.figure()
	plt.gca().set_aspect(1)
	plt.title("Original Data")
	plt.scatter(x[0], x[1], s=50, color='black', edgecolor='white')
	plt.axhline(y=0, linewidth=0.5, color='black', zorder=1)
	plt.axvline(x=0, linewidth=0.5, color='black', zorder=1)
	plt.quiver(*origin, C[:,0], C[:, 1], color='r', headlength=3.5, headaxislength=3, headwidth=3.5, width=1.3e-2, angles='xy', scale_units='xy', scale=1)
	plt.ylim(-8, 8)
	plt.xticks([-5.0, -2.5, 0, 2.5, 5.0])
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.savefig('task_1plot01.png')

	plt.figure()
	plt.gca().set_aspect(1)
	plt.title("Encoded Data")
	plt.scatter(W[0], W[1], s=50, color='black', edgecolor='white')
	plt.axhline(y=0, linewidth=0.5, color='black', zorder=1)
	plt.axvline(x=0, linewidth=0.5, color='black', zorder=1)
	plt.quiver(*origin,[0,1],[1,0], color='r', headlength=3.5, headaxislength=3, headwidth=3.5, width=3e-2, angles='xy', scale_units='xy', scale=1)
	plt.ylim(-10, 10)
	plt.xticks([-2.5, 0, 2.5])
	plt.xlabel('c1')
	plt.ylabel('c2')
	plt.savefig('task_1plot02.png')

	print("Run Task 1 Completed!")
#################### Task 2 ###################

# Implement K-Means;
# You should be able to implement your own kmeans by using numpy only.

def run_task2():
	# Loading the data
	P = 50 # Number of data points
	blobs = datasets.make_blobs(n_samples = P, centers = 3, random_state = 10)
	data = np.transpose(blobs[0])
	print(data.shape) # (2, 50)
	plt.figure()
	plt.scatter(data[0], data[1], s=50, color='black', edgecolor='white')
	plt.savefig('task_2plot01.png')
	def createCenter(K):
		return np.vstack((np.random.uniform(low=-2, high=6, size=(K,)),(np.random.uniform(low=-10, high=5, size=(K,)))))
	# TODO: fill in your code
	def kmeans(X, K, max_iter=10):

		# Initialize cluster centers
		centers = createCenter(K)
		# print(centers.shape)
		for i in range(max_iter):
			# Assign each data point to its closest cluster center
			distances = np.linalg.norm(centers[:, :, np.newaxis] - X[:, np.newaxis, :], axis=0)
			cluster_assignment = np.argmin(distances, axis=0)
			# Update cluster centers
			for j in range(K):
				centers[:, j] = centers[:, j] if j not in cluster_assignment else np.mean(X[:, cluster_assignment == j], axis=1)
		return centers
	def scree_value(X, K):
		centers = kmeans(X, K)
		distances = np.linalg.norm(centers[:, :, np.newaxis] - X[:, np.newaxis, :], axis=0)
		cluster_assignment = np.argmin(distances, axis=0)

		# Compute the total intra-cluster sum of squares
		WCSS = 0
		for j in range(K):
			WCSS += 0 if j not in cluster_assignment else np.sum(np.linalg.norm(X[:, cluster_assignment == j] - centers[:, j][:, np.newaxis], axis=0))
		return WCSS/P

	# Plot the scree value
	WCSS = []
	for i in range(1, 11):
		WCSS.append(scree_value(data, i))
	plt.figure()
	plt.gca().set_aspect(0.6)
	plt.plot(range(1, 11), WCSS, marker='o', color='black')
	plt.xlabel('number of clusters')
	plt.ylabel('cost value')
	plt.xticks(range(1, 11))
	plt.yticks(range(1,7))
	plt.savefig('task_2plot03.png')




	centers = kmeans(data, 3)
	distances = np.linalg.norm(centers[:, :, np.newaxis] - data[:, np.newaxis, :], axis=0)
	cluster_assignment = np.argmin(distances, axis=0)
	a, b, c = [], [], []
	for i in range(P):
		if cluster_assignment[i] == 0:
			a.append(data[:, i])
		elif cluster_assignment[i] == 1:
			b.append(data[:, i])
		elif cluster_assignment[i] == 2:
			c.append(data[:, i])
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
	plt.figure()
	plt.scatter(centers[0][0], centers[1][0], s=80, marker="*", color='red', edgecolor='black')
	plt.scatter(a[:, 0], a[:, 1], s=50, color='red', edgecolor='white')
	plt.scatter(centers[0][1], centers[1][1], s=80, marker="*", color='blue', edgecolor='black')
	plt.scatter(b[:, 0], b[:, 1], s=50, color='blue', edgecolor='white')
	plt.scatter(centers[0][2], centers[1][2], s=80, marker="*", color='lawngreen', edgecolor='black')
	plt.scatter(c[:, 0], c[:, 1], s=50, color='lawngreen', edgecolor='white')
	plt.legend(['centroid 1', 'cluster 1', 'centroid 2', 'cluster 2', 'centroid 3', 'cluster 3'])
	plt.savefig('task_2plot02.png')

	print("Run Task 2 Completed!")



if __name__ == '__main__':
	run_task1()
	run_task2()