from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import pdb
import json
import scipy.io
import sys
from random import shuffle
import scipy
from progressbar import ProgressBar
pbar = ProgressBar()

root_path = '../'

captions = json.load(open(os.path.join(root_path, 'captions_train2014.json'),'r'))

image_ids = [x['id'] for x in captions['images']]
image_captions = {}
for data_point in pbar(captions['annotations']):
    cur_caption = ''.join([i if ord(i) < 128 else ' ' for i in data_point['caption']])
    if data_point['image_id'] in image_captions.keys():
        image_captions[data_point['image_id']].append(cur_caption.lower().strip().replace(',','').replace('.','').replace(';',''))
    else:
        image_captions[data_point['image_id']] = [cur_caption.lower().strip().replace(',','').replace('.','').replace(';','')]

all_captions = []
for id in image_ids:
    # current_captions = {}
    # current_captions['annotations'] = map(lambda x:{'image_id':0,'caption':x}, [i.lower() for i in image_captions[id]])
    # id_captions = tokenizer.tokenize(current_captions)
    all_captions.append((image_captions[id], id))

all_data = []
for x in range(0,len(all_captions)):
    sent_combinations = list(itertools.permutations(all_captions[x][0],2))
    for combinations in sent_combinations:
        all_data.append((combinations, all_captions[x][1])) # 770876, 1574722

shuffle(all_data)

op_folder = os.path.join(root_path, 'sentence_pairs')
if not os.path.exists(op_folder):
    os.makedirs(op_folder)

f1 = open(os.path.join(op_folder,'train_enc.txt'),'w')
f2 = open(os.path.join(op_folder,'train_dec.txt'),'w')
f3 = open(os.path.join(op_folder,'train_enc_dec_ids.txt'),'w')
for j in all_data:
    a = j[0][0].replace('\n','')
    b = j[0][1].replace('\n','')
    a = ''.join([i if ord(i) < 128 else ' ' for i in a])
    b = ''.join([i if ord(i) < 128 else ' ' for i in b])
    f1.write(a)
    f1.write('\n')
    f2.write(b)
    f2.write('\n')
    f3.write(str(j[1]))
    f3.write('\n')
f1.close()
f2.close()
