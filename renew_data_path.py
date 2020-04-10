import  pickle
import pdb
import os

dirs = os.listdir('./log/')
for dir in dirs:
	files = os.listdir(os.path.join('log', dir))
	for file in files:
		if 'infos' in file:
			infos = pickle.load(open(os.path.join('log',dir,file),'rb'))
			if 'flickrbu_fc_gvd' in  infos['opt'].input_fc_dir:
				infos['opt'].dataset = 'flickr'
				infos['opt'].input_json = 'data/flickrtalk.json'
				infos['opt'].input_fc_dir = 'data/flickrbu/flickrbu_fc_gvd'
				infos['opt'].input_att_dir = 'data/flickrbu/flickrbu_att_gvd'
				infos['opt'].input_box_dir = 'data/flickrbu/flickrbu_box_gvd'
				infos['opt'].input_label_h5 = 'data/flickrtalk_label.h5'
				infos['opt'].cached_tokens  =  'flickr-train-idxs'
			elif 'flickrbu_fc'  in  infos['opt'].input_fc_dir:
				infos['opt'].dataset = 'flickr'
				infos['opt'].input_json = 'data/flickrtalk.json'
				infos['opt'].input_fc_dir = 'data/flickrbu/flickrbu_fc'
				infos['opt'].input_att_dir = 'data/flickrbu/flickrbu_att'
				infos['opt'].input_box_dir = 'data/flickrbu/flickrbu_box'
				infos['opt'].input_label_h5 = 'data/flickrtalk_label.h5'
				infos['opt'].cached_tokens  =  'flickr-train-idxs'
			elif 'cocobu_fc'  in  infos['opt'].input_fc_dir:
				infos['opt'].dataset = 'coco'
				infos['opt'].input_json = 'data/cocotalk.json'
				infos['opt'].input_fc_dir = 'data/mscoco/cocobu_fc'
				infos['opt'].input_att_dir = 'data/mscoco/cocobu_att'
				infos['opt'].input_box_dir = 'data/mscoco/cocobu_box'
				infos['opt'].input_label_h5 = 'data/cocotalk_label.h5'
				infos['opt'].cached_tokens  =  'coco-train-idxs'
			# pdb.set_trace()
			pickle.dump(infos, open(os.path.join('log',dir,file),'wb'))
