import torch
from .vocab import Vocabulary, deserialize_vocab 
import os
from .model import SCAN, get_score_attn_for_one_pair
import pdb
import nltk
import numpy as np


print('Constructing SCAN model...')
# scan_model_path='misc/SCAN/runs/f30k_scan/checkpoint/model_best.pth.tar'
scan_model_path='misc/SCAN/runs/f30k_SCAN_POS1/checkpoint/model_best.pth.tar'
# scan_model_path='misc/SCAN/runs/f30k_SCAN_t2i_AVG_POS1_gvd/checkpoint/model_best.pth.tar'
# scan_model_path='misc/SCAN/runs/f30k_SCAN_t2i_AVG_gvd/checkpoint/model_best.pth.tar'
# scan_model_path='misc/SCAN/runs/coco_SCAN/checkpoint/model_best.pth.tar'
# scan_model_path='misc/SCAN/runs/coco_SCAN_POS1/checkpoint/model_best.pth.tar'
if 'POS' in scan_model_path:
	flag = 1
else:
	flag =0 
print('scan_model_path:{}'.format(scan_model_path))
scan_data_path=None
scan_checkpoint = torch.load(scan_model_path)
scan_opt = scan_checkpoint['opt']

if scan_data_path is not None:
	scan_opt.data_path = scan_data_path

scan_opt.vocab_path='misc/SCAN/vocab'

# load vocabulary used by the model
scan_vocab = deserialize_vocab(os.path.join(scan_opt.vocab_path, '%s_vocab.json' % scan_opt.data_name))
scan_opt.vocab_size = len(scan_vocab)

# construct model
scan_model = SCAN(scan_opt)

# load model state
scan_model.load_state_dict(scan_checkpoint['model'])
scan_model.val_start()
print('Done')
# pdb.set_trace()
# scan_opt.pos = False

def get_ground_reward(images, sents, att_supervise=0, grd_shape=(145,20,36)):
	scan_opt.att_supervise=att_supervise
	if 'pos' not in scan_opt:
		scan_opt.pos = False
	# print('Computing results...')
	captions=[]
	tag_masks = []
	for sent in sents:
		tokens = sent.lower().split()
		tag = nltk.pos_tag(tokens)
		tmp_mask = []
		for t in tag:
			if t[1] in ['NN', 'NNS','NNP','NNPS'] and t[0] !='unk':
			# if t[1] in ['NN', 'NNS','NNP','NNPS','JJ','JJR','JJS'] and t[0] !='unk':
			# if t[1] in ['NN', 'NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP'] and t[0] !='unk':
				tmp_mask.append(1)
			else:
				tmp_mask.append(0)
		tmp_mask.extend([0,0])
		tmp_mask.insert(0,0)
		tag_mask = torch.BoolTensor(tmp_mask)
		# tag_mask = torch.ByteTensor(tmp_mask)
		tag_masks.append(tag_mask)

		caption = []
		caption.append(scan_vocab('<start>'))
		caption.extend([scan_vocab(token) for token in tokens])
		caption.append(scan_vocab('.'))
		caption.append(scan_vocab('<end>'))
		caption = torch.Tensor(caption)
		captions.append(caption)
	lengths=np.asarray([len(cap) for cap in captions])
	idx= np.argsort(-lengths)
	images_sort=images[idx]
	captions_sort=[]
	tag_masks_sort =[]
	for i in idx:
		captions_sort.append(captions[i])
		tag_masks_sort.append(tag_masks[i])
	assert  tag_masks_sort[0].size(0) == captions_sort[0].size(0)

	# Merget captions (convert tuple of 1D tensor to 2D tensor)
	lens = [len(cap) for cap in captions_sort]
	targets = torch.zeros(len(captions_sort), max(lens)).long()
	for i, cap in enumerate(captions_sort):
		end = lens[i]
		targets[i, :end] = cap[:end]


	# compute the embeddings
	with torch.no_grad():
		img_embs, cap_embs, cap_lens = scan_model.forward_emb(images_sort, targets, lens, volatile=True)
		if att_supervise:
			sim, att_weights = get_score_attn_for_one_pair(img_embs, cap_embs, cap_lens, scan_opt,tag_masks_sort)
		else:
			sim = get_score_attn_for_one_pair(img_embs, cap_embs, cap_lens, scan_opt,tag_masks_sort)

	score=np.zeros(len(idx))
	for i in range(len(idx)):
		score[idx[i]]=sim[i]

	if att_supervise:
		grd_weight=torch.zeros(grd_shape).to(att_weights[0].device)
		for i in range(len(idx)):
			if flag:
				# if tag_masks_sort[i][1:-2].any():
				if tag_masks_sort[i][1:-2].byte().any():
					ind = torch.nonzero(tag_masks_sort[i][1:-2]).view(-1)
					grd_weight[idx[i],ind,:] =att_weights[i]
			else:
				grd_weight[idx[i],:cap_lens[i]-3,:] =att_weights[i][1:-2]

		return score, grd_weight
	else:
		return score