import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import matplotlib
import scipy.misc

def load_image(infilename, rescale=False):
	img = Image.open(infilename)
	img.load()
	data = np.asarray(img, dtype="float32")
	# print("max = ", np.amax(data))
	if rescale:
		data /= 255.0
	return data

def change_img_ext(dump_dir, filt_dir):
	os.system('mkdir  '+ filt_dir)
	for filename in os.listdir(dump_dir):
		try:
			npa = load_image(os.path.join(dump_dir, filename))
			newfilename = ".".join(filename.split(".")[:-1])
			target_name = os.path.join(filt_dir, newfilename) + '.png'
			
			scipy.misc.imsave(target_name, npa)
		except IOError as e:
			print(e)

def change_img_ext2(dump_dir, filt_dir):
	os.system('mkdir  '+ filt_dir)
	for filename in os.listdir(dump_dir):
		try:
			npa = load_image(os.path.join(dump_dir, filename))
			im = Image.fromarray(npa)
			if im.mode != 'RGB':
				im = im.convert('RGB')
			newfilename = ".".join(filename.split(".")[:-1])
			target_name = os.path.join(filt_dir, newfilename) + '.png'
			im.save(target_name)
		except IOError as e:
			print(e)

def change_img_ext_rec(dump_dir, filt_dir):
	os.system('mkdir '+filt_dir)
	pat_len_dump = len(dump_dir)
	for root, dirlist, filelist in os.walk(dump_dir):
		if (len(filelist) == 0):
			continue
		else:
			essential_path = root[pat_len_dump:]
			dest_dir = os.path.join(filt_dir, essential_path[1:])
			os.system('mkdir ' + dest_dir)
			change_img_ext2(root, dest_dir)

change_img_ext_rec('yalefaces', 'pngyalefaces') # convert to png
