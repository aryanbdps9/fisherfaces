import helper
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA

debug = False
# debug = True
def correlation_method():
	X, y, _ = helper.load_dataset(skip_every=11)
	print("loaded dataset")
	""" For each face, get its nearest nbr(other than itself) and 
	see if it belongs to the same class
	and repeat this many_times """
	nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(X)
	raw_distances, raw_indices = nbrs.kneighbors(X)
	dists, indices = raw_distances[:,1], raw_indices[:,1]
	pred_y = y[indices]
	num_correct = np.sum(pred_y == y)
	if debug:
		printr = np.zeros((y.size, 3))
		printr[:,0]=y
		printr[:,1]=pred_y
		printr[:,2]=dists
		print("printr:\n", printr)
	print("num_correct = ", num_correct, "num_total = ", y.size)


def myPCA2(k_tot, X, drop_first_n=0, return_coeffs=True, do_plot_eigv=False):
	# X can have non-zero mean; drop_first_n means drop first n eig_vectors and eig_vals
	X_meann = np.mean(X, axis=0).reshape((1, X.shape[1]))
	X_shift = X - X_meann

	drop_condition = (drop_first_n is not None and (drop_first_n > 0))

	if drop_condition:
		k = k_tot + drop_first_n
	else:
		k = k_tot
	k = min(k, X.shape[0])
	L = np.matmul(X_shift, X_shift.T)
	eigenValues, eigenVectors = LA.eig(L)
	eigenVectors = eigenVectors.real
	# print("#"*20, "\n", eigenVectors)
	eigenValues = eigenValues.real
	idx = eigenValues.argsort()[-k:][::-1]
	eigenValues = eigenValues[idx].flatten() # k largest eigenValues
	eigenVectors = eigenVectors[:,idx]

	eigenVectors = np.matmul(X_shift.T, eigenVectors)
	eigenVectors = eigenVectors / (np.sqrt((eigenVectors*eigenVectors).sum(axis=0).reshape((1,k))))

	if drop_condition:
		eigenValues = eigenValues[drop_first_n:]
		eigenVectors = eigenVectors[:, drop_first_n:]

	# print(idx, eigenValues)
	# if do_plot_eigv:
	# 	plot_eigv(eigenValues)
	if return_coeffs:
		coeffs = np.matmul(X_shift, eigenVectors)
		return [X_meann, eigenVectors, coeffs]
	return [X_meann, eigenVectors]

"""
def myPCA3(k_tot, X, drop_first_n=0, return_coeffs=True):
	# X can have non-zero mean; drop_first_n means drop first n eig_vectors and eig_vals
	X = X.T
	X_meann = np.mean(X, axis=1).reshape((X.shape[0], 1))
	X_shift = X - X_meann

	drop_condition = (drop_first_n is not None and (drop_first_n > 0))

	if drop_condition:
		k = k_tot + drop_first_n
	else:
		k = k_tot

	L = np.matmul(X_shift.T, X_shift)
	eigenValues, eigenVectors = LA.eig(L)
	eigenVectors = np.real(eigenVectors)
	eigenValues = np.real(eigenValues)

	idx = eigenValues.argsort()[-k:][::-1]
	eigenValues = eigenValues[idx].flatten() # k largest eigenValues
	eigenVectors = eigenVectors[:,idx]

	eigenVectors = np.matmul(X_shift, eigenVectors)
	eigenVectors = eigenVectors / (np.sqrt((eigenVectors*eigenVectors).sum(axis=0).reshape((1,k))))

	if drop_condition:
		eigenValues = eigenValues[drop_first_n:]
		eigenVectors = eigenVectors[:, drop_first_n:]

	X_meann = X_meann.T
	# X_shift = X_shift.T

	if return_coeffs:
		coeffs = np.matmul(X_shift.T, eigenVectors)
		# coeffs = np.matmul(eigenVectors.T, X_shift)
		return [X_meann, eigenVectors, coeffs]
	return [X_meann, eigenVectors]
"""

def eig_face_method(drop_first_n, k_vals):
	X_raw, y_raw, _ = helper.load_dataset(skip_every=11)
	print("loaded dataset")
	ik_mat = np.zeros((X_raw.shape[0], len(k_vals)))
	k_list = np.zeros((1, len(k_vals))).flatten().tolist()
	k_vals = sorted(k_vals, reverse=True)
	# i_list = np.random.permutation(X_raw.shape[0])[:50].tolist()
	i_list = range(X_raw.shape[0])
	if debug:
		debug_mat = np.zeros((len(i_list), 3))
	for i_ind, i in enumerate(i_list):
		print("i = ", i)
		# print("(y_raw.shape) = ", (y_raw.shape))
		X = np.zeros((X_raw.shape[0]-1, X_raw.shape[1]))
		y = np.zeros((y_raw.shape[0]-1,1)).flatten()
		X[:i, :], X[i:, :] = X_raw[:i, :], X_raw[i + 1:, :]
		y[:i], y[i:] = y_raw[:i], y_raw[i+1:]
		X_meann, eigVecs, eigCoeffs = myPCA2(min(X.shape[0], k_vals[0]+1), X, drop_first_n=drop_first_n)
		X = X - X_meann
		X_test = X_raw[i,:].reshape((1, X.shape[1]))-X_meann
		eigCoeffs_test = np.matmul(X_test, eigVecs)
		for ind_k, k in enumerate(k_vals):
			# print("k, ind_k, k_vals[0], k-eigCoeffs.shape[1], eigCoeffs.shape=", k, ind_k, k_vals[0]+1, k - eigCoeffs.shape[1], eigCoeffs.shape)
			eigCoeffs_test[:, k:] = 0
			eigCoeffs[:, k:] = 0
			# print("eigCoeffs[0,:]=", eigCoeffs[0,:])
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(eigCoeffs)
			dists, indices = nbrs.kneighbors(eigCoeffs_test)
			if debug:
				debug_mat[i_ind,0] = y[indices[0,0]]
				debug_mat[i_ind,1] = y_raw[i]
				debug_mat[i_ind,2] = indices[0,0]
				# print("k = ", k, "y[indices[0,0]] = ", y[indices[0, 0]], "y_raw[i]=", y_raw[i], "indices[0,0]=", indices[0,0], "dist=", dists)
			if (y[indices[0,0]] == y_raw[i]):
				ik_mat[i,ind_k] = 1
				k_list[ind_k] += 1
			else:
				k_list[ind_k] += 0
	if debug:
		print("debug_mat\n", debug_mat)
	k_list.reverse()
	return k_list
	summary_mat = np.sum(ik_mat, axis=0).flatten().tolist()
	summary_mat.reverse()
	return summary_mat

def eig_face_driver():
	k_vals = [1, 5, 15, 40, 70, 100, 130, 150]
	# k_vals = [1, 60, 130]
	# k_vals = [1, 5, 10, 15, 25, 50, 70, 90, 125, 150]
	no_drop_eig_face = eig_face_method(0, k_vals)
	drop_eig_face = eig_face_method(3, k_vals)

	# plt.figure(figsize=(15.0, 9.0))
	plt.figure()
	plt.plot(k_vals, no_drop_eig_face)
	# plt.figure(figsize=(15.0, 9.0))
	plt.figure()
	plt.plot(k_vals, drop_eig_face)
	plt.show()

def fisher():
	X_raw, y_raw, _ = helper.load_dataset(skip_every=11)
	print("loaded dataset")
	i_list = range(X_raw.shape[0])
	for i_ind, i in enumerate(i_list):
		print("i = ", i)
		# print("(y_raw.shape) = ", (y_raw.shape))
		X = np.zeros((X_raw.shape[0]-1, X_raw.shape[1]))
		y = np.zeros((y_raw.shape[0]-1,1)).flatten()
		X[:i, :], X[i:, :] = X_raw[:i, :], X_raw[i + 1:, :]
		y[:i], y[i:] = y_raw[:i], y_raw[i+1:]
		classes, counts = np.unique(y, return_counts=True)
		freqs = dict(zip(classes, counts))
		means = []
		Xs = []
		# means = {}
		classes = classes.tolist()
		for c in classes:
			# means[c] = np.mean(X[y==c], axis=0)
			Xs.append(X[y==c])
			means.append(np.mean(Xs, axis=0))
			Xs[-1] -= means[-1]
		meann = np.mean(X, axis=0)
		mu_mat = np.concatenate(means, axis=0) - meann
		mean_wts = np.array(counts).reshape((mu_mat.shape[0], 1))
		Sb = np.matmul(mu_mat.T, mu_mat*mean_wts)
		Sw = np.zeros()
		

		
		

# correlation_method()
# eig_face_driver()
