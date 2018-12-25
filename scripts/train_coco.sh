#/bin/bash

# Define the experiment configuration
batch_size=128
num_epochs=30
base='resnet_v1_152'  # Base architecture of CNN feature extractor
cnn_weights='/shared/kgcoe-research/mil/peri/tf_checkpoints/resnet_v1_152.ckpt'  # CNN pre-trained checkpoint
lstm_weights='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/best_bleu/translate.ckpt-35000' # LSTM pre-trained checkpoint
checkpoint='/shared/kgcoe-research/mil/cvs_cvpr18/flickr_exp/c_stt-para-att_cosine_e1024_w300_L1_h1024_2018-10-23_09_33/model.ckpt-16001' # CMR model checkpoint for finetuning mode
emb_dim=1024  # CVS dimension
lr=0.0002
save_steps=2000 # Step interval for checkpoint saving
decay_steps=15000 # Steps to decay Learning rate
decay_factor=0.1 # decay factor
num_units=1024 # GRU hidden dimension
num_layers=1 # Number of layers in GRU network
dropout=0.0 # Dropout for GRU network
sim_weight=1.0 # Weight of sim loss
s2s_weight=0.25 # Weight of Seq-Seq loss
i2c_weight=0.25 # Weight of Image captioning loss
margin=0.2 # Margin for pairwise loss
word_dim=300 # Word dimension for GRU encoder
clip_grad_norm=2.0 # Gradient clipping norm value
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_stt_scan_train_r152.tfrecord' # TFRecord path to read from
record_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/f30k_stt_scan_train_r152.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/peri/flickr_data/f30k_stt_train.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/stt_data/coco_r152_train_precomp.tfrecord' # TFRecord path to read from
model='stt-att'
mode='finetune'
measure='cosine' # Type of loss
exp_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr_exp'
optimizer='adam'
dataset='mscoco'
mine_n_hard=1
lambda_1=9.0
lambda_2=6.0
# vocab_file='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc' # Size: 26375
# vocab_file='/shared/kgcoe-research/mil/cvs_cvpr18/coco/mscoco_vocab.txt'  # Size: 11355
vocab_file='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/flickr_vocab.txt'  # Size: 8483
vocab_size=8483
# vocab_size=11355
# vocab_size=26735

python ./train_stt.py --batch_size ${batch_size} \
                           --num_epochs ${num_epochs} \
                           --save_steps ${save_steps} \
                           --base ${base} \
                           --decay_steps ${decay_steps} \
                           --decay_factor ${decay_factor} \
                           --cnn_weights ${cnn_weights} \
                           --lstm_weights ${lstm_weights} \
                           --emb_dim ${emb_dim} \
                           --word_dim ${word_dim} \
                           --num_units ${num_units} \
                           --num_layers ${num_layers} \
                           --lambda_1 ${lambda_1} \
                           --lambda_2 ${lambda_2} \
                           --vocab_file ${vocab_file} \
                           --vocab_size ${vocab_size} \
                           --lr ${lr} \
                           --sim_weight ${sim_weight} \
                           --s2s_weight ${s2s_weight} \
                           --i2c_weight ${i2c_weight} \
                           --dropout ${dropout} \
                           --margin ${margin} \
                           --measure ${measure} \
                           --clip_grad_norm ${clip_grad_norm} \
                           --record_path ${record_path} \
                           --checkpoint ${checkpoint} \
                           --dataset ${dataset} \
                           --optimizer ${optimizer} \
                           --exp_path ${exp_path} \
                           --model ${model} \
                           --mine_n_hard ${mine_n_hard} \
                           --use_l2_norm \
                           --no_pretrain_lstm \
                           --precompute \
                           --no_train_cnn
                           # --mode ${mode}
                           # --finetune_with_cnn
                           # --use_same_chain \
                           # --train_only_emb
                           # --use_abs \
                           
                          
                           
                           
                           
                           
                           
                           
                          
                           
