import torch
import torch.utils.data
import numpy as np
# Assumes that tensor is (nchannels, height, width)



def tensor_RBG(x):
	return torch.stack([x[0, :, :], x[2, :, :], x[1, :, :]], 0)

def tensor_GRB(x):
	return torch.stack([x[1, :, :], x[0, :, :], x[2, :, :]], 0)

def tensor_GBR(x):
	return torch.stack([x[1, :, :], x[2, :, :], x[0, :, :]], 0)

def tensor_BRG(x):
	return torch.stack([x[2, :, :], x[0, :, :], x[1, :, :]], 0)

def tensor_BGR(x):
	return torch.stack([x[2, :, :], x[1, :, :], x[0, :, :]], 0)


def color_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		if label == 1:
			img = tensor_RBG(img)
		elif label == 2:
			img = tensor_GRB(img)
		elif label == 3:
			img = tensor_GBR(img)
		elif label == 4:
			img = tensor_BRG(img)
		elif label == 5:
			img = tensor_BGR(img)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def color_batch(batch, label):
	if label == 'rand':
		labels = torch.randint(6, (len(batch),), dtype=torch.long)
	elif label == 'expand':
		labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
					torch.zeros(len(batch), dtype=torch.long) + 1,
					torch.zeros(len(batch), dtype=torch.long) + 2,
					torch.zeros(len(batch), dtype=torch.long) + 3,
					torch.zeros(len(batch), dtype=torch.long) + 4,
					torch.zeros(len(batch), dtype=torch.long) + 5])
		batch = batch.repeat((6,1,1,1))
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return color_batch_with_labels(batch, labels), labels