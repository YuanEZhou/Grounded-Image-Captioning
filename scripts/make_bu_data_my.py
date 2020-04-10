from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse
import json
import h5py

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='/data2/yuanen/data/datasets/flickr30k', help='downloaded feature directory')
parser.add_argument('--output_dir', default='/data2/yuanen/data/datasets/flickr30k/flickrbu/flickrbu', help='output feature files')

args = parser.parse_args()


os.makedirs(args.output_dir+'_att')
os.makedirs(args.output_dir+'_fc')
os.makedirs(args.output_dir+'_box')

with open(os.path.join(args.downloaded_feats, 'dic_flickr30k.json')) as f:
    dic=json.load(f)
    img=dic['images']

feat=h5py.File(os.path.join(args.downloaded_feats, 'flickr_det_feat.hdf5'),'r')


cnt=0
for i in range(len(img)):
    print('{}/{}'.format(cnt,31783))
    np.savez_compressed(os.path.join(args.output_dir+'_att', str(img[i]['id'])), feat=feat['image_features'][i])
    np.save(os.path.join(args.output_dir+'_fc', str(img[i]['id'])), feat['image_features'][i].mean(0))
    np.save(os.path.join(args.output_dir+'_box', str(img[i]['id'])), feat['image_bb'][i])
    cnt+=1




