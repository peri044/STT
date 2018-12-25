#/bin/bash

# Define the experiment configuration
batch_size=1
num=5000
base='resnet_v2_152'
checkpoint='/shared/kgcoe-research/mil/Flickr30k/c_vse_order_e2048_w1024_L2_h1024_2018-08-31_13_18/model.ckpt-8000'
emb_dim=2048
num_units=1024
num_layers=2
dropout=0.0
word_dim=1024
model='vse'
val_ids_path='/shared/kgcoe-research/mil/Flickr30k/flickr30k_images'
val_caps_path='/shared/kgcoe-research/mil/peri/mscoco_data/train_caps.txt'
record_path='/shared/kgcoe-research/mil/peri/mscoco_data/coco_val_precompute.tfrecord'
measure='order'
root_path='/shared/kgcoe-research/mil/Flickr30k/flickr30k_images/flickr30k_images'
dataset='flickr'
precompute = Ture

python eval.py --batch_size ${batch_size} \
                 --num ${num} \
                 --base ${base} \
                 --checkpoint ${checkpoint} \
                 --emb_dim ${emb_dim} \
                 --word_dim ${word_dim} \
                 --num_units ${num_units} \
                 --num_layers ${num_layers} \
                 --dropout ${dropout} \
                 --measure ${measure} \
                 --val_ids_path ${val_ids_path} \
                 --root_path ${root_path} \
                 --val_caps_path ${val_caps_path} \
                 --model ${model} \
                 --dataset ${dataset} \
                 --use_abs\
                 --precompute ${precompute}
                 
                           