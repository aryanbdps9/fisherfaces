import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io
import h5py
import cv2
import sys
if (sys.version_info[0] == 3):
    from functools import reduce


def mean_normalise(X):
    meann = np.mean(X, axis=0)
    X = X - meann
    return X

def variance_normalise(X, newvar=1.0):
    varr = np.var(X, axis=0)
    X /= varr
    X[X==np.inf] = 0
    X[X == -np.inf] = 0
    res = np.nan_to_num(X)
    return res

def mv_normalise(X, newvar=1.0):
    return variance_normalise(mean_normalise(X), newvar=newvar)

def make_fig(nparr, cmap=None):
	if (cmap is None):
		plt.imshow(nparr)
	else:
		plt.imshow(nparr, cmap=cmap)
	# plt.colorbar()

def rescale_mat(npa):
    minn, maxx = np.amin(npa), np.amax(npa)
    if (minn != maxx):
        npa = (npa - minn) / (maxx - minn)
        return npa
    elif (minn >= 0 and maxx <= 1):
        return npa
    elif (minn >= 0 and maxx <= 255):
        return (npa / 255.0)
    else:
        print("unable_to_rescale")
        return np.zeros(npa.shape)

def load_image(infilename, normalise=True):
	img = Image.open(infilename)
	img.load()
	data = np.asarray(img, dtype="float32")
	# print("max = ", np.amax(data))
	if normalise:
		data = mv_normalise(data)
	return data

def load_dataset(src_dir='./pngyalefaces', num_classes=15, shape=None, skip_every=11):
    # returns X, classes and poses
    filenames=sorted(os.listdir(src_dir))
    filenames = [os.path.join(src_dir, filename) for filename in filenames]
    if shape is None:
        shape = load_image(filenames[0], normalise=False).shape
    size = reduce(lambda x, y: x*y, shape)
    res = np.zeros((len(filenames), size))
    resy = []
    cnt = 0
    ctrl = 0
    for file in filenames:
        if (skip_every != 0 and ctrl % skip_every == 0):
            ctrl += 1
            continue
        rect_img = load_image(file, normalise=False)
        res[cnt, :] = rect_img.flatten()
        resy.append(cnt)
        cnt += 1
        ctrl += 1
    # print("res.shape = ", res.shape)
    res = res[:cnt, :]
    res = mv_normalise(res)
    resy = np.array(resy)
    row_per_class = cnt / num_classes
    # if (skip_every != 0):
    #     row_per_class -= 1
    classes = np.floor(resy / row_per_class)
    poses = resy % row_per_class
    # print("load_dataset: shapes: res, classes, poses::", res.shape, classes.shape, poses.shape)
    return res, classes, poses

def read_mat(fname, as_np_array=True, rescale01 = True, get_keys=False):
    # function to read a .mat file and return the list of objects in that file
    res = []
    key_names = []
    try:
        f = h5py.File(fname, 'r')
        for key in list(f.keys()):
            if (as_np_array):
                if not rescale01:
                    res.append(np.array(f.get(key), dtype='float32'))
                else:
                    res.append(rescale_mat(np.array(f.get(key), dtype='float32')))
            else:
                res.append(f.get(key))
            key_names.append(key)
    except (OSError):
        f = scipy.io.loadmat(fname) # f will be a dictionary
        for key in f:
            if (key == '__header__' or key == '__version__' or key == '__globals__'):
                continue
            # print("key################## = ", key)
            if (as_np_array):
                if not rescale01:
                    res.append(np.array(f[key], dtype='float32'))
                else:
                    res.append(rescale_mat(np.array(f[key], dtype='float32')))
            else:
                res.append(f[key])
            key_names.append(key)
    if (get_keys):
        return [res, key_names]
    return res

# sample usage:
# data = read_mat('../data/grassNoisy.mat')[0]
# print(data.shape)


def noisify(inp, noise_type, val):
    imshape = inp.shape
    if (noise_type == 'gauss'):
        # val is sigma
        print('enter gauss')
        noise = np.random.normal(0, val, imshape)
        return np.maximum(np.minimum(inp + noise, 1.0), 0.0)

    if (noise_type == 'salt_and_pepper' or noise_type == 'sap' or noise_type == 's&p'):
        # val is fraction of corrupted pixels
        p_by_2 = val / 2.0
        randmat = np.random.random(imshape)
        zero_mask_inv = np.logical_not(randmat < p_by_2)
        one_mask = randmat > (1 - p_by_2)
        out_img = np.multiply(inp, zero_mask_inv)
        return np.maximum(out_img, one_mask)
