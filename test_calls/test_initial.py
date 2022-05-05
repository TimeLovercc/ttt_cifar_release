from __future__ import print_function
import argparse

import torch
from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

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
parser.add_argument('--grad_corr', action='store_true')
parser.add_argument('--visualize_samples', action='store_true')
########################################################################
parser.add_argument('--outf', default='.')
parser.add_argument('--resume', default=None)
parser.add_argument('--none', action='store_true')

args = parser.parse_args()
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, head_color, ssh, ssh_color = build_model(args)
teset, teloader = prepare_test_data(args)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load(args.resume + '/ckpt.pth')
net.load_state_dict(ckpt['net'])
cls_initial, cls_correct, cls_losses = test(teloader, net)

print('Old test error cls %.2f' %(ckpt['err_cls']*100))
print('New test error cls %.2f' %(cls_initial*100))

if args.none:
	rdict = {'cls_initial': cls_initial, 'cls_correct': cls_correct, 'cls_losses': cls_losses}
	torch.save(rdict, args.outf + '/%s_%d_none.pth' %(args.corruption, args.level))
	quit()

print('Old test error ssh %.2f' %(ckpt['err_ssh']*100))
head.load_state_dict(ckpt['head'])
head_color.load_state_dict(ckpt['head_color'])
ssh_initial, ssh_correct, ssh_losses = [], [], []
ssh_initial2, ssh_correct2, ssh_losses2 = [], [], []

labels = [0,1,2,3]
labels2 = [0,1,2,3,4,5]
for label in labels:
	tmp = test(teloader, ssh, ssh_color, sslabel=label)
	ssh_initial.append(tmp[0])
	ssh_initial2.append(tmp[1])
	ssh_correct.append(tmp[2])
	ssh_correct2.append(tmp[3])
	ssh_losses.append(tmp[4])
	ssh_losses2.append(tmp[5])

rdict = {'cls_initial': cls_initial, 'cls_correct': cls_correct, 'cls_losses': cls_losses,
			'ssh_initial': ssh_initial, 'ssh_correct': ssh_correct, 'ssh_losses': ssh_losses,
			'ssh_initial2': ssh_initial2, 'ssh_correct2': ssh_correct2, 'ssh_losses2': ssh_losses2}
torch.save(rdict, args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))

if args.grad_corr:
	corr, corr2 = test_grad_corr(teloader, net, ssh, ssh_color, ext)
	print('Average gradient inner product: %.2f' %(mean(corr)))
	torch.save((corr, corr2), args.outf + '/%s_%d_grc.pth' %(args.corruption, args.level))