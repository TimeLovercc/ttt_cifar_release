import numpy as np
import torch
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('colorblind')

# corruptions_names = ['original', 'gauss', 'shot', 'impulse', 'defocus', 'glass']

# corruptions_names_short = ['orig', 'gauss', 'shot', 'impul', 'defoc', 'glass']
corruptions_names_short = ['gauss', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom', 
							'snow', 'frost', 'fog', 'bright', 'contra', 'elastic', 'pixel', 'jpeg']
# corruptions_names.insert(0, 'orig')

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

info = []
info.append(('gn', '_expand', 5))


########################################################################

def easy_barplot(table, fname, name, width=0.3):
	## changed
	labels = ['Baseline', 'Joint training', 'Test-time training', "Joint training mulit", "Test-time training mulit", "Joint training personalized", "Test-time training personalized"]
	index =  np.asarray(range(len(table[0,:])))

	plt.figure(figsize=(5, 4))
	for i, row in enumerate(table):
		# plt.bar(index + i*(width*2), row, width)
		plt.bar(labels[i], row, width)

	plt.ylabel('Error (%)')
	# changed
	# plt.xticks(index + width/4, corruptions_names[i+1])
	# end
	plt.title(name)
	plt.xticks(rotation=45)
	plt.legend(prop={'size': 8})
	plt.tight_layout(pad=0)
	plt.savefig(fname)
	plt.close()

def easy_latex(table, prec1=True):
	for row in table:
		row_str = ''
		for entry in row:
			if prec1:
				row_str += '& %.1f' %(entry)
			else:
				row_str += '& %s' %(entry)
		print(row_str)

for parta, partb, level in info:
	print(level, parta + partb)
	results = torch.load('results/C10C_layer2_%d_%s%s.pth' %(level, parta, partb))
	col = results.shape[1]
	for i in range(col):
		easy_barplot(np.expand_dims(results[:,i], axis=1), 'results/C10C_layer2_%d_%s%s_%d.pdf' %(level, parta, partb, i), corruptions_names_short[i+1])
		# easy_latex([corruptions_names_short], prec1=False)
		# easy_latex(results[:,i])
		
