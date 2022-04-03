import numpy as np
import torch
import torch.nn as nn
from utils.misc import *
from utils.rotation import rotate_batch
from utils.color import color_batch
def build_model(args):
	from models.ResNet import ResNetCifar as ResNet
	from models.SSHead import ExtractorHead
	print('Building model...')
	if args.dataset[:7] == 'cifar10':
		classes = 10
	elif args.dataset == 'cifar7':
		if not hasattr(args, 'modified') or args.modified:
			classes = 7
		else:
			classes = 10

	if args.group_norm == 0:
		norm_layer = nn.BatchNorm2d
	else:
		def gn_helper(planes):
			return nn.GroupNorm(args.group_norm, planes)
		norm_layer = gn_helper

	net = ResNet(args.depth, args.width, channels=3, classes=classes, norm_layer=norm_layer).cuda()
	if args.shared == 'none':
		args.shared = None
	if args.shared == 'layer3' or args.shared is None:
		from models.SSHead import extractor_from_layer3
		ext = extractor_from_layer3(net)
		head = nn.Linear(64 * args.width, 4)
		head_color = nn.Linear(64 * args.width, 6)
	elif args.shared == 'layer2':
		from models.SSHead import extractor_from_layer2, head_on_layer2
		ext = extractor_from_layer2(net)
		head = head_on_layer2(net, args.width, 4)
		head_color = nn.Linear(64 * args.width, 6)
	ssh = ExtractorHead(ext, head).cuda()
	ssh_color = ExtractorHead(ext, head_color).cuda()
	if hasattr(args, 'parallel') and args.parallel:
		net = torch.nn.DataParallel(net)
		ssh = torch.nn.DataParallel(ssh)
		ssh_color = torch.nn.DataParallel(ssh_color)
	return net, ext, head, ssh,ssh_color
	

def test(dataloader, model,model_color, sslabel=None):
	criterion = nn.CrossEntropyLoss(reduction='none').cuda()
	model.eval()
	correct = []
	correct_color = []
	losses = []
	losses_color = []
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		if sslabel is not None:
			inputs, labels = rotate_batch(inputs, sslabel)
			color_inputs, color_labels = color_batch(inputs, sslabel)
		inputs, labels = inputs.cuda(), labels.cuda()
		color_inputs, color_labels = color_inputs.cuda(), color_labels.cuda()
		with torch.no_grad():
			outputs = model(inputs)
			outputs_color=model_color(color_inputs)
			loss = criterion(outputs, labels)
			loss_color=criterion(outputs_color, color_labels)
			losses.append(loss.cpu())
			losses_color.append(loss_color.cpu())
			_, predicted = outputs.max(1)
			_, predicted_color = outputs_color.max(1)
			correct.append(predicted.eq(labels).cpu())
			correct_color.append(predicted_color.eq(color_labels).cpu())
	correct = torch.cat(correct).numpy()
	correct_color = torch.cat(correct_color).numpy()
	losses = torch.cat(losses).numpy()
	losses_color = torch.cat(losses_color).numpy()
	model.train()
	return 1-correct.mean(),1-correct_color.mean(), correct, losses,losses_color

def test_grad_corr(dataloader, net, ssh, ext):
	criterion = nn.CrossEntropyLoss().cuda()
	net.eval()
	ssh.eval()
	corr = []
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		net.zero_grad()
		ssh.zero_grad()
		inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
		outputs_cls = net(inputs_cls)
		loss_cls = criterion(outputs_cls, labels_cls)
		grad_cls = torch.autograd.grad(loss_cls, ext.parameters())
		grad_cls = flat_grad(grad_cls)
		ext.zero_grad()
		inputs, labels = rotate_batch(inputs, 'expand')
		inputs_ssh, labels_ssh = inputs.cuda(), labels.cuda()
		outputs_ssh = ssh(inputs_ssh)
		loss_ssh = criterion(outputs_ssh, labels_ssh)
		grad_ssh = torch.autograd.grad(loss_ssh, ext.parameters())
		grad_ssh = flat_grad(grad_ssh)

		corr.append(torch.dot(grad_cls, grad_ssh).item())
	net.train()
	ssh.train()
	return corr


def pair_buckets(o1, o2):
	crr = np.logical_and( o1, o2 )
	crw = np.logical_and( o1, np.logical_not(o2) )
	cwr = np.logical_and( np.logical_not(o1), o2 )
	cww = np.logical_and( np.logical_not(o1), np.logical_not(o2) )
	return crr, crw, cwr, cww
def count_each(tuple):
	return [item.sum() for item in tuple]


def plot_epochs(all_err_cls, all_err_ssh, fname, use_agg=True):
	import matplotlib.pyplot as plt
	if use_agg:
		plt.switch_backend('agg')

	plt.plot(np.asarray(all_err_cls)*100, color='r', label='classifier')
	plt.plot(np.asarray(all_err_ssh)*100, color='b', label='self-supervised')
	plt.xlabel('epoch')
	plt.ylabel('test error (%)')
	plt.legend()
	plt.savefig(fname)
	plt.close()
