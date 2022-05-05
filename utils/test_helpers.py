import numpy as np
import torch
import torch.nn as nn
from utils.misc import *
from utils.rotation import rotate_batch
from utils.color import color_batch
def build_model(args, training=False):
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
		head_personal = nn.Linear(64 * args.width, 2)
	elif args.shared == 'layer2':
		from models.SSHead import extractor_from_layer2, head_on_layer2
		ext = extractor_from_layer2(net)
		head = head_on_layer2(net, args.width, 4)
		head_color = head_on_layer2(net, args.width, 6)
		head_personal = head_on_layer2(net, args.width, 2)
	ssh = ExtractorHead(ext, head).cuda()
	ssh_color = ExtractorHead(ext, head_color).cuda()
	ssh_personal = ExtractorHead(ext, head_personal).cuda()
	if hasattr(args, 'parallel') and args.parallel:
		net = torch.nn.DataParallel(net)
		ssh = torch.nn.DataParallel(ssh)
		ssh_color = torch.nn.DataParallel(ssh_color)
		ssh_personal = torch.nn.DataParallel(ssh_personal)
	if training:
		return net, ext, head, head_color, head_personal, ssh,ssh_color, ssh_personal
	else:
		return net, ext, head, head_color, ssh,ssh_color
	

def test(dataloader, model,model_color=None, sslabel=None):
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
			color_inputs, color_labels = color_inputs.cuda(), color_labels.cuda()

		inputs, labels = inputs.cuda(), labels.cuda()
		with torch.no_grad():
			outputs = model(inputs)
			if sslabel is not None:
				outputs_color=model_color(color_inputs)
				loss_color=criterion(outputs_color, color_labels)
				losses_color.append(loss_color.cpu())
				_, predicted_color = outputs_color.max(1)
				correct_color.append(predicted_color.eq(color_labels).cpu())

			loss = criterion(outputs, labels)
			losses.append(loss.cpu())
			_, predicted = outputs.max(1)
			correct.append(predicted.eq(labels).cpu())
	correct = torch.cat(correct).numpy()
	losses = torch.cat(losses).numpy()
	if sslabel is not None:
		correct_color = torch.cat(correct_color).numpy()
		losses_color = torch.cat(losses_color).numpy()
	model.train()
	if sslabel is None:
		return 1-correct.mean(), correct, losses
	return 1-correct.mean(),1-correct_color.mean(), correct, correct_color, losses,losses_color

def test_grad_corr(dataloader, net, ssh, ssh_color, ext):
	criterion = nn.CrossEntropyLoss().cuda()
	net.eval()
	ssh.eval()
	corr = []
	corr2 = []
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
		# print(grad_cls.shape, grad_ssh.shape)
		# print(torch.dot(grad_cls.detach(), grad_ssh.detach()))
		corr.append(np.dot(grad_cls.cpu().numpy(), grad_ssh.cpu().numpy()))

		inputs, labels = color_batch(inputs, 'expand')
		inputs_ssh, labels_ssh = inputs.cuda(), labels.cuda()
		outputs_ssh = ssh_color(inputs_ssh)
		loss_ssh = criterion(outputs_ssh, labels_ssh)
		grad_ssh = torch.autograd.grad(loss_ssh, ext.parameters())
		grad_ssh = flat_grad(grad_ssh)

		corr2.append(np.dot(grad_cls.cpu().numpy(), grad_ssh.cpu().numpy()))

	net.train()
	ssh.train()
	ssh_color.train()
	return corr, corr2


def pair_buckets(o1, o2):
	crr = np.logical_and( o1, o2 )
	crw = np.logical_and( o1, np.logical_not(o2) )
	cwr = np.logical_and( np.logical_not(o1), o2 )
	cww = np.logical_and( np.logical_not(o1), np.logical_not(o2) )
	return crr, crw, cwr, cww
def count_each(tuple):
	return [item.sum() for item in tuple]


def plot_epochs(all_err_cls, all_err_ssh, all_err_ssh_color, fname, use_agg=True):
	import matplotlib.pyplot as plt
	if use_agg:
		plt.switch_backend('agg')

	plt.plot(np.asarray(all_err_cls)*100, color='r', label='classifier')
	plt.plot(np.asarray(all_err_ssh)*100, color='b', label='self-supervised-rotation')
	plt.plot(np.asarray(all_err_ssh_color)*100, color='b', label='self-supervised-color')
	plt.xlabel('epoch')
	plt.ylabel('test error (%)')
	plt.legend()
	plt.savefig(fname)
	plt.close()
