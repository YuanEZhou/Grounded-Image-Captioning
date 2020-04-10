import os
import pdb
dirs = os.listdir('./log/')
for dir in dirs:
	# pdb.set_trace()
	files = os.listdir('./log/'+dir)
	for file in files:
		if '-best' not in file  or 'optimizer' in file:
			os.remove(os.path.join('log',dir,file))