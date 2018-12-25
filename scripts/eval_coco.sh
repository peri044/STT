#/bin/bash

# Define the experiment configuration
batch_size=100
num=25000
base='resnet_v1_152'
checkpoint=$1
emb_dim=1024
num_units=1024
num_layers=1
num_folds=5
dropout=0.0
word_dim=300
model='stt'
val_ids_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/flickr_scan_test_image_ids.txt'
# val_caps_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/flickr_scan_test_sentences.txt'
val_caps_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/flickr_scan_test_sentences.txt'
test_sample='1007129816.jpg'
# val_ids_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/test_filenames.txt'
# val_caps_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/testall_caps.txt'
# test_sample='COCO_val2014_000000436141.jpg'
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/feats/rv1_152_tf/coco_new_test.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/flickr_data/flickr_test_r152_precomp.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/scan_data/f30k_test_scan.tfrecord'
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_scan_test.tfrecord'    
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/para_att_pred.tfrecord'    
record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_dual_r152_test.tfrecord'
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/f30k_scan_test_r152.tfrecord'
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/f30k_test_dual.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/scan_data/coco_train_scan.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/im_dual_tfrecords/mscoco_val.tfrecord'
measure='cosine'
root_path='shared/kgcoe-research/mil/peri/mscoco_data'
mode='val'
# vocab_file='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc'
# vocab_file='/shared/kgcoe-research/mil/cvs_cvpr18/coco/mscoco_vocab.txt'
vocab_file='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/flickr_vocab.txt'  # Size: 8481
# vocab_size=11355
# vocab_size=26735
vocab_size=8483

python ./eval_gpu.py --batch_size ${batch_size} \
                 --base ${base} \
                 --checkpoint ${checkpoint} \
                 --emb_dim ${emb_dim} \
                 --num ${num} \
                 --num_folds ${num_folds} \
                 --word_dim ${word_dim} \
                 --num_units ${num_units} \
                 --vocab_file ${vocab_file} \
                 --vocab_size ${vocab_size} \
                 --num_layers ${num_layers} \
                 --dropout ${dropout} \
                 --measure ${measure} \
                 --val_ids_path ${val_ids_path} \
                 --test_sample ${test_sample} \
                 --record_path ${record_path} \
                 --root_path ${root_path} \
                 --val_caps_path ${val_caps_path} \
                 --model ${model} \
                 --mode ${mode} \
                 --precompute
                 # --retrieve_text
                 # --use_abs
                           