import helper
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
import scipy.linalg as SLA

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
	return num_correct*100.0 / y.size, y.size



def myPCA4(k_tot, X, return_coeffs=True, drop_first_n=0):
	X_meann = np.mean(X, axis=0).reshape((1, X.shape[1]))
	X_shift = X - X_meann
	L = np.matmul(X_shift, X_shift.T)
	eigVal, eigVec = LA.eig(L)
	eigVec = eigVec[:, :k_tot]
	eigVal = eigVal[:k_tot]
	eigVec = np.matmul(X_shift.T, eigVec)
	eigVec = eigVec.real
	eigVec /= np.sqrt(np.sum(eigVec*eigVec, axis=0)).reshape((1, k_tot))
	drop_condition = (drop_first_n is not None and (drop_first_n > 0))
	if drop_condition:
		eigVal = eigVal[drop_first_n:]
		eigVec = eigVec[:, drop_first_n:]
	if (return_coeffs):
		coeffs = np.matmul(X_shift, eigVec)
		return [X_meann, eigVec, coeffs]
	else:
		return [X_meann, eigVec]


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
def fisher3(X_raw, y_raw):
	# X_raw, y_raw, _ = helper.load_dataset(skip_every=11)
	i_list = range(X_raw.shape[0])
	num_correct = 0
	for i_ind, i in enumerate(i_list):
		print("i = ", i)
		# print("(y_raw.shape) = ", (y_raw.shape))
		X = np.zeros((X_raw.shape[0]-1, X_raw.shape[1]))
		y = np.zeros((y_raw.shape[0]-1,1)).flatten()
		X[:i, :], X[i:, :] = X_raw[:i, :], X_raw[i+1:, :]
		# X_meann = np.mean(X, axis=0)
		y[:i], y[i:] = y_raw[:i], y_raw[i+1:]
		# take pca, mean_overall, note eigenvec, coeffs
		classes, counts = np.unique(y, return_counts=True)
		cl_list = classes.tolist()
		num_classes = len(cl_list)
		if (i_ind == 0):
			print("num_classes:", num_classes)
		X_meann, PCAeigvec, Xcoeffs = myPCA4(X_raw.shape[0]-num_classes, X, return_coeffs=True)
		PCAdmeans = np.mean(Xcoeffs, axis=0)
		# compute class means in pca'd space
		means = []
		Xs = []
		for c in classes:
			Xs.append(Xcoeffs[y == c])
			means.append(np.expand_dims(np.mean(Xs[-1], axis=0), axis=0))
			Xs[-1] -= means[-1]
		mu_mat = np.concatenate(means, axis=0) - PCAdmeans
		mean_wts = np.array(counts).reshape((len(means), 1))
		wtd_mu_mat = mu_mat * mean_wts
		# compute sb, sw
		Sb = np.matmul(mu_mat.T, wtd_mu_mat)
		Sw = np.zeros(Sb.shape)
		for c_ind, c in enumerate(classes):
			Sw += np.matmul(Xs[c_ind].T, Xs[c_ind])
		# compute lda_eigvecs
		LDAeigvals, LDAeigvecs = SLA.eig(Sb, Sw)
		LDAeigvecs = LDAeigvecs.real
		# compute ldb_ev = pca_ev * lda_eigvecs
		final_lda_ev = np.matmul(PCAeigvec, LDAeigvecs)
		# find means of all classes
		mu_mat += PCAdmeans
		mu_mat = np.matmul(mu_mat, LDAeigvecs)
		# find X in ldb space:
		X_transformed = np.matmul(Xcoeffs, LDAeigvecs)
		# keep only: ldb_ev, mean_overall, means of classes
		# i.e. final_lda_ev, X_meann, mu_mat

		# testing:
		# subtract mean_overall
		X_test = X_raw[i, :].reshape((1, X.shape[1])) - X_meann
		# transform using ldb_ev
		transformed_test = np.matmul(X_test, final_lda_ev)
		# find nearest neighbor with means of classes
		nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(X_transformed)
		# nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(mu_mat)
		dists, indices = nbrs.kneighbors(transformed_test)
		if (y_raw[i] == y[indices[0,0]]):
			print("i = ", i, ": correct")
			num_correct += 1
		else:
			print("i = ", i, ": wrong")
	return num_correct

		# predict that class, and check if correct


def fisher2(X_raw, y_raw):
	# X_raw, y_raw, _ = helper.load_dataset(skip_every=11)
	i_list = range(X_raw.shape[0])
	num_correct = 0
	for i_ind, i in enumerate(i_list):
		print("i = ", i)
		# print("(y_raw.shape) = ", (y_raw.shape))
		X = np.zeros((X_raw.shape[0]-1, X_raw.shape[1]))
		y = np.zeros((y_raw.shape[0]-1,1)).flatten()
		X[:i, :], X[i:, :] = X_raw[:i, :], X_raw[i+1:, :]
		# X_meann = np.mean(X, axis=0)
		y[:i], y[i:] = y_raw[:i], y_raw[i+1:]
		# take pca, mean_overall, note eigenvec, coeffs
		classes, counts = np.unique(y, return_counts=True)
		cl_list = classes.tolist()
		num_classes = len(cl_list)
		X_meann, PCAeigvec, Xcoeffs = myPCA4(X_raw.shape[0]-num_classes, X, return_coeffs=True)
		PCAdmeans = np.mean(Xcoeffs, axis=0)
		# compute class means in pca'd space
		means = []
		Xs = []
		for c in classes:
			Xs.append(Xcoeffs[y == c])
			means.append(np.expand_dims(np.mean(Xs[-1], axis=0), axis=0))
			Xs[-1] -= means[-1]
		mu_mat = np.concatenate(means, axis=0) - PCAdmeans
		mean_wts = np.array(counts).reshape((len(means), 1))
		wtd_mu_mat = mu_mat * mean_wts
		# compute sb, sw
		Sb = np.matmul(mu_mat.T, wtd_mu_mat)
		Sw = np.zeros(Sb.shape)
		for c_ind, c in enumerate(classes):
			Sw += np.matmul(Xs[c_ind].T, Xs[c_ind])
		# compute lda_eigvecs
		LDAeigvals, LDAeigvecs = SLA.eig(Sb, Sw)
		LDAeigvecs = LDAeigvecs.real
		# compute ldb_ev = pca_ev * lda_eigvecs
		final_lda_ev = np.matmul(PCAeigvec, LDAeigvecs)
		# find means of all classes
		mu_mat += PCAdmeans
		mu_mat = np.matmul(mu_mat, LDAeigvecs)

		# keep only: ldb_ev, mean_overall, means of classes
		# i.e. final_lda_ev, X_meann, mu_mat

		# testing:
		# subtract mean_overall
		X_test = X_raw[i, :].reshape((1, X.shape[1])) - X_meann
		# transform using ldb_ev
		transformed_test = np.matmul(X_test, final_lda_ev)
		# find nearest neighbor with means of classes
		nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(mu_mat)
		dists, indices = nbrs.kneighbors(transformed_test)
		if (y_raw[i] == indices[0,0]):
			print("i = ", i, ": correct")
			num_correct += 1
		else:
			print("i = ", i, ": wrong")
	return num_correct

		# predict that class, and check if correct

def fisher_driver():
	X_raw, y_raw, poses = helper.load_dataset(skip_every=11)
	print("fisher2(face reco:):", fisher2(X_raw, y_raw))
	# print("fisher2(pose estimation):", fisher3(X_raw, poses))

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
		# X_meann, eigVecs, eigCoeffs = myPCA2(min(X.shape[0], k_vals[0]+1), X, drop_first_n=drop_first_n)
		X_meann, eigVecs, eigCoeffs = myPCA4(min(X.shape[0], k_vals[0]+1), X, drop_first_n=drop_first_n)
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

def linear():
	X_raw, y_raw, z_raw = helper.load_dataset(skip_every=11) ###3, 5, 6
	classes, counts = np.unique(y_raw, return_counts=True)
	num_classes = classes.shape[0]
	img_size = X_raw.shape[1]
	num_img = X_raw.shape[0]
	bases = np.zeros((num_classes, 3, img_size))
	X_norm = X_raw/np.sqrt(np.sum(X_raw*X_raw, axis=1)).reshape(num_img,1)
	# bases1 = X_raw[z_raw == 3]
	# bases2 = X_raw[z_raw == 5]
	# bases3 = X_raw[z_raw == 6]

	indx = [2, 3, 5]
	rep_indx = [6,4,6]
	# rep_indx = [2,3,5]

	bases[:,0,:] = X_norm[z_raw == indx[0]]
	bases[:,1,:] = X_norm[z_raw == indx[1]]
	bases[:,2,:] = X_norm[z_raw == indx[2]]

	for i in range(num_classes):
		bases[i,1,:] = bases[i,1,:] - (np.matmul(bases[i,1,:], bases[i,0,:].T))*bases[i,0,:]
		bases[i,1,:] /= np.sqrt(np.sum(bases[i,1,:]*bases[i,1,:]))
		
		bases[i,2,:] = bases[i,2,:] - (np.matmul(bases[i,2,:], bases[i,0,:].T))*bases[i,0,:]
		bases[i,2,:] = bases[i,2,:] - (np.matmul(bases[i,2,:], bases[i,1,:].T))*bases[i,1,:]
		# bases[i,0,:] /= np.sqrt(np.sum(bases[i,0,:]*bases[i,0,:]))
		bases[i,2,:] /= np.sqrt(np.sum(bases[i,2,:]*bases[i,2,:]))
	

	correct = 0
	for i in range(num_img):
		pose = int(z_raw[i])
		# print(pose)
		class1 = int(y_raw[i])
		restore_base = np.zeros((3, img_size))
		restore_base[:,:] = bases[class1,:,:]
		if(pose == indx[0]):
			continue
			bases[class1,0,:] = X_norm[i+(rep_indx[0]-pose)]
			
			bases[class1,1,:] = X_norm[i+(indx[1]-pose)]
			bases[class1,1,:] = bases[class1,1,:] - (np.matmul(bases[class1,1,:], bases[class1,0,:].T))*bases[class1,0,:]
			bases[class1,1,:] /= np.sqrt(np.sum(bases[class1,1,:]*bases[class1,1,:]))
			
			bases[class1,2,:] = X_norm[i+(indx[2]-pose)]
			bases[class1,2,:] = bases[class1,2,:] - (np.matmul(bases[class1,2,:], bases[class1,0,:].T))*bases[class1,0,:]
			bases[class1,2,:] = bases[class1,2,:] - (np.matmul(bases[class1,2,:], bases[class1,1,:].T))*bases[class1,1,:]
			bases[class1,2,:] /= np.sqrt(np.sum(bases[class1,2,:]*bases[class1,2,:]))
			# bases[class1,0,:] /= np.sqrt(np.sum(bases[class1,0,:]*bases[class1,0,:]))
		elif(pose == indx[1]):
			# continue
			bases[class1,1,:] = X_norm[i+(rep_indx[1]-pose)]
			bases[class1,1,:] = bases[class1,1,:] - (np.matmul(bases[class1,1,:], bases[class1,0,:].T))*bases[class1,0,:]
			bases[class1,1,:] /= np.sqrt(np.sum(bases[class1,1,:]*bases[class1,1,:]))
			
			bases[class1,2,:] = X_norm[i+(indx[2]-pose)]
			bases[class1,2,:] = bases[class1,2,:] - (np.matmul(bases[class1,2,:], bases[class1,0,:].T))*bases[class1,0,:]
			bases[class1,2,:] = bases[class1,2,:] - (np.matmul(bases[class1,2,:], bases[class1,1,:].T))*bases[class1,1,:]
			bases[class1,2,:] /= np.sqrt(np.sum(bases[class1,2,:]*bases[class1,2,:]))
			# bases[class1,0,:] /= np.sqrt(np.sum(bases[class1,0,:]*bases[class1,0,:]))
		elif(pose == indx[2]):
			continue
			# print("base0 = ",bases[class1, 0,:])
			# print("base1 = ",bases[class1, 1,:])
			bases[class1,2,:] = X_norm[i+(rep_indx[2]-pose)]
			# print("base2 = ",bases[class1, 2,:])
			bases[class1,2,:] = bases[class1,2,:] - (np.matmul(bases[class1,2,:], bases[class1,0,:].T))*bases[class1,0,:]
			# print("base2 = ",bases[class1, 2,:])
			bases[class1,2,:] = bases[class1,2,:] - (np.matmul(bases[class1,2,:], bases[class1,1,:].T))*bases[class1,1,:]
			# print("base2 = ",bases[class1, 2,:])
			# print("i = ",i, " class1 = ",class1, " pose = ", pose, " val = ", np.sqrt(np.sum(bases[class1,2,:]*bases[class1,2,:])))
			bases[class1,2,:] /= np.sqrt(np.sum(bases[class1,2,:]*bases[class1,2,:]))
			# bases[class1,1,:] = bases[class1,1,:] - (np.matmul(bases[class1,1,:], bases[class1,0,:].T))*bases[class1,0,:]
			# bases[class1,0,:] /= np.sqrt(np.sum(bases[class1,0,:]*bases[class1,0,:]))
			# bases[class1,1,:] /= np.sqrt(np.sum(bases[class1,1,:]*bases[class1,1,:]))


		img = X_raw[i]
		min_val = -1
		ans = -1
		for j in range(num_classes):
			img1 = img.T
			obt_img = (np.matmul(bases[j,0,:], img1))*bases[j,0,:] + (np.matmul(bases[j,1,:],img1))*bases[j,1,:] + (np.matmul(bases[j,2,:],img1))*bases[j,2,:]
			diff = obt_img - img
			diff_val = np.sqrt(np.sum(diff*diff))
			if(min_val != -1):
				if(min_val > diff_val):
					min_val = diff_val
					ans = j
			else:
				min_val = diff_val
				ans = j

		if(ans == y_raw[i]):
			correct+=1
		bases[class1,:,:] = restore_base


	print("LinSub: Correct Predictions = ", correct, "num_total = ", y_raw.size)
	return (correct * 100.0 / y_raw.size), y_raw.size





	# np.sqrt(np.sum(eigVec*eigVec, axis=0)).reshape((1, k_tot))
	# bases /= np.sqrt(np.sum(bases*bases, axis=2)).reshape(num_classes, 3)

	# print(bases.shape)



def fisher():
	X_raw, y_raw, _ = helper.load_dataset(skip_every=11)
	_, _, X_raw = myPCA4(X_raw.shape[0]-15, X_raw, return_coeffs=True)
	print("loaded dataset")
	i_list = range(X_raw.shape[0])
	num_correct = 0
	for i_ind, i in enumerate(i_list):
		print("i = ", i)
		# print("(y_raw.shape) = ", (y_raw.shape))
		X = np.zeros((X_raw.shape[0]-1, X_raw.shape[1]))
		y = np.zeros((y_raw.shape[0]-1,1)).flatten()
		X[:i, :], X[i:, :] = X_raw[:i, :], X_raw[i+1:, :]
		X_meann = np.mean(X, axis=0)
		y[:i], y[i:] = y_raw[:i], y_raw[i+1:]
		classes, counts = np.unique(y, return_counts=True)
		means = []
		Xs = []
		# means = {}
		classes = classes.tolist()
		for c in classes:
			# means[c] = np.mean(X[y==c], axis=0)
			Xs.append(X[y==c])
			means.append(np.expand_dims(np.mean(Xs[-1], axis=0), axis=0))
			Xs[-1] -= means[-1]
		meann = np.expand_dims(np.mean(X, axis=0), axis=0)
		# [print(Xselem.shape) for Xselem in Xs]
		# [print(term.shape) for term in means]
		# print(meann.shape)
		mu_mat = np.concatenate(means, axis=0) - meann
		mean_wts = np.array(counts).reshape((mu_mat.shape[0], 1))
		wtd_mu_mat = mu_mat * mean_wts
		# print("wtd_mu_mat.shape = ", wtd_mu_mat.shape)
		Sb = np.matmul(mu_mat.T, wtd_mu_mat)
		Sw = np.zeros((X.shape[1], X.shape[1]))
		# print("Sw.shape = ", Sw.shape)
		for c_ind, c in enumerate(classes):
			Sw += np.matmul(Xs[c_ind].T, Xs[c_ind])
		eigvals, eigvecs = SLA.eig(Sb, Sw)#, eigvals_only=False)
		# eigvals, eigvecs = LA.eig(np.matmul(LA.inv(Sw), Sb))
		# eigvals, eigvecs = LA.eig(np.matmul(LA.inv(Sw), Sb))
		eigvecs = eigvecs.real[:,:len(classes)-1]
		# print("eigvals\n", eigvals)
		eigCoeffs = np.matmul(X-X_meann, eigvecs)
		mean_eigCoeffs = np.zeros((len(classes), len(classes)-1))
		# for mean_ind in
		# print("eigvecs.shape")
		X_test = X_raw[i, :].reshape((1, X.shape[1])) - X_meann
		eigCoeffs_test = np.matmul(X_test, eigvecs)
		nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(eigCoeffs)
		dists, indices = nbrs.kneighbors(eigCoeffs_test)
		if (y[indices[0,0]] == y_raw[i]):
			num_correct += 1
		# print(eigvals)
		# break
		# eigvals, eigvecs = SLA.eigh(Sb, Sw, eigvals_only=False)
		# print("eigvecs.shape = ", eigvecs.shape)
	return num_correct


# print("fisher2() =", fisher2())
# print("fisher2() =", fisher2())
# fisher_driver()
# correlation_method()
# linear()
# eig_face_driver()
# print("fisher() =", fisher())
def main():
	correlation_accu, num_total = correlation_method()
	# k_vals = [1, 5, 15, 40, 70, 100, 130, 150]
	# k_vals = [1, 60, 130]
	k_vals = [1, 5, 10, 15, 25, 50, 70, 90, 125, 150]
	no_drop_eig_face = [x * 100.0 / num_total for x in eig_face_method(0, k_vals)]
	drop_eig_face = [x * 100.0 / num_total for x in eig_face_method(3, k_vals)]
	linear_accu, _ = linear()
	X_raw, y_raw, _ = helper.load_dataset(skip_every=11)
	fisher_accu = fisher2(X_raw, y_raw) * 100.0 / num_total
	# plt.figure(figsize=(15.0, 9.0))
	fig = plt.figure()
	plt.plot(k_vals, no_drop_eig_face, color='C1', label="Eigenfaces")
	plt.plot(k_vals, drop_eig_face, color='C2', label="Eigenfaces w/o first 3 components")
	plt.axhline(y=fisher_accu, color='C3', label="Fisherfaces")
	plt.axhline(y=linear_accu, color='C4', label="Linear Subspace")
	plt.axhline(y=correlation_accu, color='C5', label="Correlation")
	plt.gca().legend(loc='lower right')  # show legend
	# plt.figure(figsize=(15.0, 9.0))
	fig.savefig("my_plot.png")
	# plt.figure()
	plt.show()


main()