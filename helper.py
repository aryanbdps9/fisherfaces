import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io
import h5py
import cv2


def load_image(infilename, rescale=True):
	img = Image.open(infilename)
	img.load()
	data = np.asarray(img, dtype="float32")
	# print("max = ", np.amax(data))
	if rescale:
		data /= 255.0
	return data

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
