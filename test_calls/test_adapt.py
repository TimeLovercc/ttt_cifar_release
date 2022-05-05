from __future__ import print_function
import argparse
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import *
from utils.color import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', default='/data/yusun/datasets/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--fix_bn', action='store_true')
parser.add_argument('--fix_ssh', action='store_true')
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--outf', default='.')
parser.add_argument('--resume', default=None)

args = parser.parse_args()
args.threshold += 0.001		# to correct for numeric errors
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, head_color,head_personal, ssh,ssh_color, ssh_personal = build_model(args, training=True)
teset, teloader = prepare_test_data(args)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load(args.resume + '/ckpt.pth')
if args.online:
	net.load_state_dict(ckpt['net'])
	head.load_state_dict(ckpt['head'])
	head_color.load_state_dict(ckpt['head_color'])
	head_personal.load_state_dict(ckpt['head_personal'])

criterion_ssh = nn.CrossEntropyLoss(reduction='none').cuda()
if args.fix_ssh:
	optimizer_ssh = optim.SGD(ext.parameters(), lr=args.lr)
else:
	optimizer_ssh = optim.SGD(list(ssh.parameters())+list(ssh_color.parameters()) + list(ssh_personal.parameters()), lr=args.lr)

def adapt_single(image):
	if args.fix_bn:
		ssh.eval()
		ssh_color.eval()
		ssh_personal.eval()
	elif args.fix_ssh:
		ssh.eval()
		ssh_color.eval()
		ssh_personal.eval()
		ext.train()
	else:
		ssh.train()
		ssh_color.train()
		ssh_personal.train()
	for iteration in range(args.niter):
		inputs = [tr_transforms(image) for _ in range(args.batch_size)]
		inputs = torch.stack(inputs).cuda()
		outputs_ssh_personal = ssh_personal(inputs)
		outputs_ssh_personal = F.softmax(outputs_ssh_personal, dim=1)

		inputs_ssh, labels_ssh = rotate_batch(inputs, 'rand')
		inputs_ssh_color, labels_ssh_color = color_batch(inputs, 'rand')
		inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
		inputs_ssh_color, labels_ssh_color = inputs_ssh_color.cuda(), labels_ssh_color.cuda()
		optimizer_ssh.zero_grad()
		outputs_ssh = ssh(inputs_ssh)
		outputs_ssh_color = ssh_color(inputs_ssh_color)
		loss_ssh = criterion_ssh(outputs_ssh, labels_ssh)
		loss_ssh_color = criterion_ssh(outputs_ssh_color, labels_ssh_color)
		loss_ssh = loss_ssh.reshape(inputs.shape[0], -1).sum(dim=1)
		loss_ssh_color = loss_ssh_color.reshape(inputs.shape[0], -1).sum(dim=1)
		loss = outputs_ssh_personal[:,0] * loss_ssh + outputs_ssh_personal[:,1] * loss_ssh_color
		loss_ssh = loss.mean()
		loss_ssh.backward()
		optimizer_ssh.step()

def test_single(model, model2, image, label):
	model.eval()
	inputs = te_transforms(image).unsqueeze(0)
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)
		confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()
	correctness = 1 if predicted.item() == label else 0

	model2.eval()
	inputs = te_transforms(image).unsqueeze(0)
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)
		confidence2 = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()
	correctness2 = 1 if predicted.item() == label else 0

	correctness = (correctness+correctness2)/2
	confidence = (confidence+confidence2)/2
	return correctness, confidence

def trerr_single(model, model2, image):
	model.eval()
	labels = torch.LongTensor([0, 1, 2, 3])
	inputs = torch.stack([te_transforms(image) for _ in range(4)])
	inputs = rotate_batch_with_labels(inputs, labels)
	inputs, labels = inputs.cuda(), labels.cuda()
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)

	model.eval()
	labels2 = torch.LongTensor([0, 1, 2, 3, 4, 5])
	inputs = torch.stack([te_transforms(image) for _ in range(6)])
	inputs = color_batch_with_labels(inputs, labels2)
	inputs, labels2 = inputs.cuda(), labels2.cuda()
	with torch.no_grad():
		outputs = model2(inputs.cuda())
		_, predicted2 = outputs.max(1)

	return predicted.eq(labels).cpu(), predicted2.eq(labels2).cpu()

print('Running...')
correct = []
sshconf = []
trerror = []
trerror2 = []
if args.dset_size == 0:
	args.dset_size = len(teset)
for i in tqdm(range(1, args.dset_size+1)):
	if not args.online:
		net.load_state_dict(ckpt['net'])
		head.load_state_dict(ckpt['head'])
		head_color.load_state_dict(ckpt['head_color'])
		head_personal.load_state_dict(ckpt['head_personal'])

	_, label = teset[i-1]
	image = Image.fromarray(teset.data[i-1])

	sshconf.append(test_single(ssh, ssh_color, image, 0)[1])
	if sshconf[-1] < args.threshold:
		adapt_single(image)
	correct.append(test_single(net, ssh_color, image, label)[0])
	error, error2 = trerr_single(ssh, ssh_color,image)
	trerror.append(error)
	trerror2.append(error2)

rdict = {'cls_correct': np.asarray(correct), 'ssh_confide': np.asarray(sshconf), 
		'cls_adapted':1-mean(correct), 'trerror': trerror, 'terror2':trerror2}
torch.save(rdict, args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))
