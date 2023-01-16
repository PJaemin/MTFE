import os
import sys
import torchvision
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
import glob
import random
import Metrics
import cv2
import time

random.seed(123)





class input_loader(data.Dataset):

	def __init__(self, image_path):
		self.image_path = image_path
		self.in_files = self.list_files(os.path.join(image_path, 'input'))


	def data_augment(self, inp, gt):
		a = random.randint(1, 4)
		if a == 1:
			return inp, gt
		elif a == 2:
			return inp.rotate(180, expand=True), gt.rotate(180, expand=True)
		elif a == 3:
			return ImageOps.flip(inp.rotate(180, expand=True)), ImageOps.flip(gt.rotate(180, expand=True))
		else:
			return ImageOps.flip(inp), ImageOps.flip(gt)

	def __getitem__(self, index):
		fname = os.path.split(self.in_files[index])[-1]
		data_low = Image.open(self.in_files[index])
		data_gt = Image.open(os.path.join(self.image_path, 'gt', fname))

		low = np.asarray(data_low)
		data_hist = np.zeros((3, 256))
		for i in range(3):
			S = low[..., i]
			data_hist[i, ...], _ = np.histogram(S.flatten(), 256, [0, 256])
			data_hist[i, ...] = data_hist[i, ...] / np.sum(data_hist[i, ...])

		data_input, data_gt = self.data_augment(data_low, data_gt)

		data_input = (np.asarray(data_input)/255.0)
		data_gt = (np.asarray(data_gt)/255.0)

		data_input = torch.from_numpy(data_input).float()
		data_gt = torch.from_numpy(data_gt).float()
		data_hist = torch.from_numpy(data_hist).float()


		return data_input.permute(2,0,1), data_gt.permute(2,0,1), data_hist

	def __len__(self):
		return len(self.in_files)

	def list_files(self, in_path):
		files = []
		for (dirpath, dirnames, filenames) in os.walk(in_path):
			files.extend(filenames)
			break
		files = sorted([os.path.join(in_path, x) for x in files])
		return files
