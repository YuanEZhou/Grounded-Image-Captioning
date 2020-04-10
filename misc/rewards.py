from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
import nltk
import pdb

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.spice.spice import Spice
from .SCAN.ground_reward import get_ground_reward

CiderD_scorer = None
Bleu_scorer = None
Spice_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global Spice_scorer
    Spice_scorer = Spice_scorer or Spice()

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)
    
    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        if opt['att_supervise']:
            greedy_res, _, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
        else:
            greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
    model.train()

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    if opt['cider_reward_weight'] > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        # print('Cider scores:', _)
    else:
        cider_scores = 0

    if opt['bleu_reward_weight'] > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    if opt['spice_reward_weight']>0:
        spice_gts = {}
        spice_res__ = {}
        for k,v in gts.items():
            tmp=[]
            for v_i in v:
                tmp_sent = utils.decode_sequence(opt['vocab'], np.asarray(list(map(int,v_i.split()))).reshape(1,-1))
                tmp.extend(tmp_sent)
            spice_gts[k] = tmp

        for k,v in res__.items():
            spice_res__[k] = utils.decode_sequence(opt['vocab'], np.asarray(list(map(int,v[0].split()))).reshape(1,-1))

        _, spice_scores = Spice_scorer.compute_score(spice_gts, spice_res__)
        tmp_score = []
        for i in spice_scores:
            tmp_score.append(i['All']['f'])
        spice_scores = np.asarray(tmp_score)
    else:
        spice_scores = 0


    if opt['ground_reward_weight']>=0:
        att_feats_repeat = att_feats.repeat(2,1,1)
        gen_sents = utils.decode_sequence(opt['vocab'], gen_result)
        greedy_sents = utils.decode_sequence(opt['vocab'], greedy_res)
        sents = gen_sents+greedy_sents
        if opt['att_supervise']:
            ground_scores, grd_weights = get_ground_reward(att_feats_repeat, sents, opt['att_supervise'],(len(sents),greedy_res.shape[1],att_feats.shape[1]))
        else:
            ground_scores = get_ground_reward(att_feats_repeat, sents, opt['att_supervise'],(len(sents),greedy_res.shape[1],att_feats.shape[1]))
        # print('Ground scores:', ground_scores.mean())
    else:
        ground_scores=0

    scores = opt['cider_reward_weight'] * cider_scores + opt['bleu_reward_weight'] * bleu_scores+ opt['ground_reward_weight']*ground_scores  +  opt['spice_reward_weight']*spice_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    if opt['att_supervise']:
        valid_mask = torch.zeros(gen_result.shape)
        for i in range(gen_result.shape[0]):
            tokens = nltk.pos_tag(gen_sents[i].split())
            for j in range(len(tokens)):
                if tokens[j][1] in ['NN', 'NNS','NNP','NNPS'] and tokens[j][0] !='unk':
                    valid_mask[i,j]=1
        return rewards, grd_weights[:batch_size], valid_mask
    else:
        return rewards
