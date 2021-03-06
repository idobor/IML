import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np

def plot_vector_as_image(image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title('title', size=12)
	plt.show()

def get_pictures_by_name(name='George W Bush'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)
	return selected_images, h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	# Compute PCA on the given matrix.

	# Args:
	# 	X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
	# 	For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
	# 	k - number of eigenvectors to return

	# Returns:
	#   U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
	#   		of the covariance matrix.
	#   S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
  U = None
  S = None
  n = len(X)
  X_tag = np.sum(X,axis=0)/n
  for i,x in enumerate(X):
    X[i]-=X_tag
  
  cov = np.matmul(X.T,X) #* (1/(n))
  eigen_values, eigen_vectors = np.linalg.eig(cov)
  eigen_vectors = eigen_vectors.T
  eigen_tuples = [(eigen_values[i],eigen_vectors[i])  for i in range(len(eigen_values))]
  eigen_tuples.sort(reverse=True)
  U = np.array([eigen_tuples[i][1] for i in range(k)]) 
  S = np.array([eigen_tuples[i][0] for i in range(k)]) 
  return U,S

def task_b():
  selected_images, h, w = get_pictures_by_name()
  X = np.squeeze(np.array(selected_images))
  U,S = PCA(X,10)
  for u in U:
    plot_vector_as_image(u,h,w)


def task_c():
  selected_images, h, w = get_pictures_by_name()
  n = len(selected_images)
  K = [1,5,10,30,50,100]
  D = []
  X = np.squeeze(np.array(selected_images))
  for k in K:
    dist = 0
    print(k)
    U,S = PCA(X,k) # shape U is(k,d)
    X_pca = np.dot(X,U.T) # shape (n,k)
    X_transformed = np.dot(X_pca,U) # shape (n,d)
    rand_indices = np.random.randint(n, size=5)
    for rand in rand_indices:
      plot_vector_as_image(selected_images[rand],h,w)
      plot_vector_as_image(X_transformed[rand],h,w)
      dist+=np.linalg.norm(X[rand]-X_transformed[rand])
      print(dist)
    D.append(dist)
  plt.plot(K,D)
  plt.xlabel("k")
  plt.ylabel("distance sum")
  plt.show()
#task_b()
task_c()



