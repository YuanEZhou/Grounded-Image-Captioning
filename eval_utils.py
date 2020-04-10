from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import pdb
from collections import defaultdict
from eval_grd_flickr30k_entities import FlickrGrdEval
from misc.rewards import init_scorer, get_self_critical_reward


bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr' in dataset:
        annFile = 'coco-caption/annotations/caption_flickr30k.json'
    else:
        raise
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    # pdb.set_trace()
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out
 
def eval_split(model, crit, loader, eval_kwargs={}):
    eval_att = eval_kwargs.get('eval_att',False)
    gt_grd_eval = eval_kwargs.get('gt_grd_eval',False)
    eval_scan = eval_kwargs.get('eval_scan',False)
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    grd_output = defaultdict(list)

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'],data['box_feats']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks, box_feats = tmp

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels, att_masks)[0], labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        if not gt_grd_eval:
            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
                data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                data['box_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, box_feats, att_masks = tmp
        else:
            tmp = [data['fc_feats'], 
                data['att_feats'],
                data['box_feats'],
                data['att_masks'] if data['att_masks'] is not None else None]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, box_feats, att_masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if eval_att:
                if not gt_grd_eval:
                    assert eval_kwargs['beam_size']==1,  'only support beam_size is 1'
                    seq, _, att_weights = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')
                    seq=seq.detach()
                    att_weights=att_weights.detach()
                    att_ind = torch.max(att_weights, dim=2)[1]
                else:
                    if not eval_scan:
                        #==This snippet used for evaluating grounding accuracy of caption model on gt sentence.=====#
                        _, att_weights=model(fc_feats, att_feats, labels, att_masks)
                        seq = labels[:,1:]
                        att_weights=att_weights.detach()
                        att_ind = torch.max(att_weights, dim=2)[1]
                        data['infos'] = data['infos']*5
                    else:
                        # pdb.set_trace()
                        #====This snippet used for evaluating grounding accuracy of SCAN model on gt sentence.======#
                        gts = data['gts']
                        reward, att_weights, noun_mask= get_self_critical_reward(model, fc_feats, att_feats, att_masks, gts, labels[:,1:], eval_kwargs)
                        seq =  labels[:,1:]
                        att_weights=att_weights.detach()
                        att_ind = torch.max(att_weights, dim=2)[1]
                        data['infos'] = data['infos']*5

                for i in range(seq.size(0)):
                    tmp_result = {'clss':[], 'idx_in_sent':[], 'bbox':[]}
                    num_sent = 0 # does not really matter which reference to use
                    for j in range(seq.size(1)):
                        if seq[i,j].item() != 0:
                            lemma = loader.wtol[loader.ix_to_word[str(seq[i,j].item())]]
                            if lemma in loader.lemma_det_dict:
                                tmp_result['bbox'].append(box_feats[i, att_ind[i, j], :4].tolist())
                                tmp_result['clss'].append(loader.itod[loader.lemma_det_dict[lemma]])
                                tmp_result['idx_in_sent'].append(j) # redundant, for the sake of output format
                        else:
                            break
                    grd_output[str(data['infos'][i]['id'])].append(tmp_result)
            else:
                seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data

        
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        if not gt_grd_eval:
            lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    if eval_att:
        # write attention results to file
        attn_file = 'att_results/attn-gen-sent-results-'+split+'-'+eval_kwargs['id']+'.json'
        with open(attn_file, 'w') as f:
            json.dump({'results':grd_output, 'eval_mode':'gen', 'external_data':{'used':True, 'details':'Object detector pre-trained on Visual Genome on object detection task.'}}, f)

        # offline eval
        evaluator = FlickrGrdEval(reference_file=eval_kwargs['reference'], submission_file=attn_file,
                              split_file=eval_kwargs['split_file'], val_split=[split],
                              iou_thresh=0.5)

        print('\nResults Summary (generated sent):')
        print('Printing attention accuracy on generated sentences...')
        if not gt_grd_eval:
            prec_all, recall_all, f1_all = evaluator.grd_eval(mode='all')
            prec_loc, recall_loc, f1_loc = evaluator.grd_eval(mode='loc')
        else:
            grd_accu = evaluator.gt_grd_eval()
        print('\n')


    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
