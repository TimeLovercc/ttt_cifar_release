import sys
import numpy as np
import torch
from utils.misc import *
from test_calls.show_result import get_err_adapted

# corruptions_names = ['gauss', 'shot', 'impulse', 'defocus', 'glass', 'motion']
# # corruptions_names.insert(0, 'orig')

# corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur']
# corruptions.insert(0, 'original')

corruptions_names = ['gauss', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom', 
							'snow', 'frost', 'fog', 'bright', 'contra', 'elastic', 'pixel', 'jpeg']
# corruptions_names.insert(0, 'orig')

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']



info = []
info.append(('gn', '_expand', 5))

baseline = []
for level in [5]:
	baseline += [('', '', level)]


########################################################################

def print_table(table, prec1=True):
	for row in table:
		row_str = ''
		for entry in row:
			if prec1:
				row_str += '%.1f\t' %(entry)
			else:
				row_str += '%s\t' %(str(entry))
		print(row_str)

def show_table(folder, level, threshold):
	results = []
	for corruption in corruptions:
		row = []

		rdict_ada = torch.load(folder + '/%s_%d_ada.pth' %(corruption, level))
		rdict_inl = torch.load(folder + '/%s_%d_inl.pth' %(corruption, level))

		ssh_confide = rdict_ada['ssh_confide']
		new_correct = rdict_ada['cls_correct']
		old_correct = rdict_inl['cls_correct']

		row.append(rdict_inl['cls_initial'])
		old_correct = old_correct[:len(new_correct)]
		err_adapted = get_err_adapted(new_correct, old_correct, ssh_confide, threshold=threshold)
		row.append(err_adapted)

		results.append(row)

	results = np.asarray(results)
	results = np.transpose(results)
	results = results * 100
	return results

def show_table_our(folder, level, threshold):
	results = []
	for corruption in corruptions:
		row = []

		rdict_ada = torch.load(folder + '/%s_%d_ada_color.pth' %(corruption, level))
		rdict_inl = torch.load(folder + '/%s_%d_inl_color.pth' %(corruption, level))

		ssh_confide = rdict_ada['ssh_confide']
		new_correct = rdict_ada['cls_correct']
		old_correct = rdict_inl['cls_correct']

		row.append(rdict_inl['cls_initial'])
		old_correct = old_correct[:len(new_correct)]
		err_adapted = get_err_adapted(new_correct, old_correct, ssh_confide, threshold=threshold)
		row.append(err_adapted)

		results.append(row)

	results = np.asarray(results)
	results = np.transpose(results)
	results = results * 100
	return results

def show_table_person(folder, level, threshold):
	results = []
	for corruption in corruptions:
		row = []

		rdict_ada = torch.load(folder + '/our_new/%s_%d_ada.pth' %(corruption, level))
		rdict_inl = torch.load(folder + '/our_new/%s_%d_inl.pth' %(corruption, level))

		ssh_confide = rdict_ada['ssh_confide']
		new_correct = rdict_ada['cls_correct']
		old_correct = rdict_inl['cls_correct']

		row.append(rdict_inl['cls_initial'])
		old_correct = old_correct[:len(new_correct)]
		err_adapted = get_err_adapted(new_correct, old_correct, ssh_confide, threshold=threshold)
		row.append(err_adapted)

		results.append(row)

	results = np.asarray(results)
	results = np.transpose(results)
	results = results * 100
	return results

def show_none(folder, level):
	results = []
	for corruption in corruptions:
		try:
			rdict_inl = torch.load(folder + '/%s_%d_none.pth' %(corruption, level))
			results.append(rdict_inl['cls_initial'])
		except:
			results.append(0)
	results = np.asarray([results])
	results = results * 100
	return results

for parta, partb, level in info:
	print(level, parta + partb)
	print_table([corruptions_names], prec1=False)
	if parta == 'bn':
		threshold = 0.9
	else:
		threshold = 1

	results_none = show_none('results/C10C_none_%s_%s' %('none', parta), level)
	print_table(results_none)

	results_slow = show_table('results/C10C_layer2_%s_%s%s' %('slow', parta, partb), level, threshold=threshold)
	print_table(results_slow)

	results_multi = show_table_our('results/C10C_layer2_%s_%s%s' %('slow', parta, partb), level, threshold=threshold)
	print_table(results_multi)

	results_person = show_table_person('results/C10C_layer2_%s_%s%s' %('slow', parta, partb), level, threshold=threshold)
	print_table(results_person)

	# results_onln = show_table('results/C10C_layer2_%s_%s%s' %('online', parta, partb), level, threshold=threshold)
	# results_onln = results_onln[1:,:]
	# print_table(results_onln)

	results = np.concatenate((results_none, results_slow, results_multi, results_person))
	torch.save(results, 'results/C10C_layer2_%d_%s%s.pth' %(level, parta, partb))

for parta, partb, level in baseline:
	if parta == '':
		print(level)
		print_table([corruptions_names], prec1=False)
		continue
	results_none = show_none('results/C10C_none_baseline_%s_bl_%s' %(parta, partb), level)
	print_table(results_none)
