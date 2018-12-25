#/bin/bash

# Define the experiment configuration
batch_size=128
num_epochs=50
base='resnet_v2_152'  # Base architecture of CNN feature extractor
cnn_weights='/shared/kgcoe-research/mil/peri/tf_checkpoints/resnet_v2_152.ckpt'  # CNN pre-trained checkpoint
lstm_weights='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/best_bleu/translate.ckpt-35000' # LSTM pre-trained checkpoint
checkpoint='/shared/kgcoe-research/mil/Flickr30k/c_vse_order_e2048_w1024_L2_h1024_2018-08-31_11_49/model.ckpt-2000' # CMR model checkpoint for finetuning mode
emb_dim=2048  # CVS dimension
lr=0.001
save_steps=2000 # Step interval for checkpoint saving
num_units=1024 # GRU hidden dimension
num_layers=2 # Number of layers in GRU network
dropout=0.0 # Dropout for GRU network
margin=0.05 # Margin for pairwise loss
word_dim=1024 # Word dimension for GRU encoder
clip_grad_norm=1.0 # Gradient clipping norm value
record_path='/shared/kgcoe-research/mil/Flickr30k/flickr30k_resnet_train.tfrecord' # TFRecord path to read from
model='vse'
mode='finetune'
measure='order' # Type of loss
exp_path='/shared/kgcoe-research/mil/Flickr30k'
optimizer='adam'
dataset='flickr30k'
precompute = True

python /home/pxu4114/CVS/cvs_cmr/cvs/CMR/train_crossmodal.py --batch_size ${batch_size} \
                           --num_epochs ${num_epochs} \
                           --save_steps ${save_steps} \
                           --base ${base} \
                           --cnn_weights ${cnn_weights} \
                           --lstm_weights ${lstm_weights} \
                           --emb_dim ${emb_dim} \
                           --word_dim ${word_dim} \
                           --num_units ${num_units} \
                           --num_layers ${num_layers} \
                           --lr ${lr} \
                           --dropout ${dropout} \
                           --margin ${margin} \
                           --measure ${measure} \
                           --clip_grad_norm ${clip_grad_norm} \
                           --record_path ${record_path} \
                           --dataset ${dataset} \
                           --optimizer ${optimizer} \
                           --exp_path ${exp_path} \
                           --model ${model} \
                           --mode ${mode} \
                           --use_abs \
                           --checkpoint ${checkpoint}\
                           --no_train_cnn\
                           --precompute ${precompute}
