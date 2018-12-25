import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import argparse
from model import *
from skimage import io
from skimage.transform import resize
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from preprocessing import preprocessing_factory
from data.coco_data_loader import *
import pdb
import time
# tf.enable_eager_execution()
    
def order_sim_gpu(images_placeholder, text_placeholder):
    """
    Computes the order similarity between images and captions
    """
    clip_diff = tf.maximum(tf.subtract(text_placeholder, images_placeholder), 0)    
    sqr_clip_diff = tf.square(clip_diff)
    sim = tf.sqrt(tf.reduce_sum(sqr_clip_diff, axis=-1))
    sim = -tf.transpose(sim)
    
    return sim 

def t2i_gpu(image_embeddings, text_embeddings, measure='order'):
    """
    Text-Image retrieval on GPU (much faster compared to CPU impl. Refer to legacy for cpu imp)
    Args: 
        image_embeddings: 5000 x emb_dim
        text_embeddings: 5000 x emb_dim
    Returns: 
         Recall scores and ranks
    """
    # Runs a batch of 50 text samples with all other image embeddings in the dataset
    # Tiling to replicate each text sample to match number of total image samples
    text_tensor = tf.placeholder(shape=(50, text_embeddings.shape[1]), dtype=tf.float32)
    image_tensor = tf.placeholder(shape=(image_embeddings.shape[0]/5, image_embeddings.shape[1]), dtype=tf.float32)

    text_exp_tensor = tf.expand_dims(text_tensor, 1)
    tile_text_embeddings =  tf.tile(text_exp_tensor, [1, image_embeddings.shape[0]/5, 1], name='tile_text_embeddings')

    image_exp_tensor = tf.expand_dims(image_tensor, 0)  
    tile_image_embeddings = tf.tile(image_exp_tensor, [50, 1, 1], name='tile_image_embeddings')
    
    if measure=='order':
        d = order_sim_gpu(tile_image_embeddings,tile_text_embeddings)
        inds = tf.contrib.framework.argsort(d,direction="DESCENDING",axis=0)
    
    inds_np=np.zeros((text_embeddings.shape[0], 1000), dtype=np.int32)
    # Unique image embeddings in the 5000 replicated original image_embeddings
    unique_im_embeddings = image_embeddings[0:5000:5]
    
    if measure=='order':
        with tf.Session() as sess:
            for i in range(0, inds_np.shape[0], 50):
                idx = sess.run(inds, feed_dict={image_tensor:unique_im_embeddings,
                                          text_tensor: text_embeddings[i:i+50]})
                inds_np[i: i+50, :] = idx.T
    elif measure=='cosine':
        sim_scores = np.matmul(text_embeddings, unique_im_embeddings.T)
        inds_np = np.argsort(sim_scores)[:, ::-1] 
        
    ranks = np.zeros(inds_np.shape[0])
    top1 = np.zeros(inds_np.shape[0])

    for i in range(0, 1000):
        for index in range(5):
            ranks[5 * i + index] = np.where(inds_np[5 * i + index] == i)[0][0]
            top1[5 * i + index] = inds_np[5 * i + index][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # R@1
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)  # R@5
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks) # R@10

    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1), inds_np

def i2t_gpu(image_embeddings, text_embeddings, measure='order'):
    """
    Image-Text retrieval on GPU (much faster compared to CPU impl. Refer to legacy for cpu imp)
    Args: 
        image_embeddings: 5000 x emb_dim
        text_embeddings: 5000 x emb_dim
    Returns: 
         Recall scores and ranks
    """
    # Runs a batch of 50 image samples with all other text embeddings in the dataset
    # Tiling to replicate each image sample to match number of total image samples
    text_tensor = tf.placeholder(shape=(image_embeddings.shape[0], image_embeddings.shape[1]), dtype=tf.float32)
    image_tensor = tf.placeholder(shape=(50, image_embeddings.shape[1]), dtype=tf.float32)
    
    text_exp_tensor = tf.expand_dims(text_tensor, 0)
    tile_text_embeddings =  tf.tile(text_exp_tensor, [50, 1, 1], name='tile_text_embeddings')
    
    image_exp_tensor = tf.expand_dims(image_tensor, 1)  
    tile_image_embeddings = tf.tile(image_exp_tensor, [1, image_embeddings.shape[0], 1], name='tile_image_embeddings')
    
    if measure=='order':
        d = order_sim_gpu(tile_image_embeddings, tile_text_embeddings)
        inds = tf.contrib.framework.argsort(d,direction="DESCENDING",axis=0)

    unique_im_emb = image_embeddings[0:image_embeddings.shape[0]:5,:]
    inds_np=np.zeros((unique_im_emb.shape[0], image_embeddings.shape[0]), dtype=np.int32)
    
    if measure=='order':
        with tf.Session() as sess:
            for i in range(0, unique_im_emb.shape[0], 50):
                idx = sess.run(inds, feed_dict={image_tensor:unique_im_emb[i: i+50],
                                          text_tensor: text_embeddings})
                inds_np[i: i+50, :] = idx.T
    elif measure=='cosine':
        sim_scores = np.matmul(unique_im_emb, text_embeddings.T)
        inds_np = np.argsort(sim_scores)[:, ::-1]   
        
    ranks = np.zeros(unique_im_emb.shape[0], dtype=np.int32)
    top1 = np.zeros(unique_im_emb.shape[0], dtype=np.int32)
    
    for i in range(inds_np.shape[0]):
        rank = 1e20
        for index in range(5*i, 5*i + 5, 1):
            tmp = np.where(inds_np[i] == index)[0][0]   # Actual GT indices are 10*index to 10*index +5. tmp is the rank of these items.
            if tmp < rank:
                rank = tmp
        ranks[i] = rank
        top1[i] = inds_np[i][0]
        
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # R@1
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)  # R@5
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks) # R@10
    
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return (r1, r5, r10, medr, meanr), (ranks, top1), inds_np 
    
	
def eval(args):

    dataset = CocoDataLoader(precompute=args.precompute, model=args.model)
    image, caption, seq_len = dataset._read_data(args.record_path, args.batch_size, phase=args.mode, num_epochs=args.num_epochs)

    # Call Show, Translate and Tell model
    model=STT(base=args.base, margin=args.margin, embedding_dim=args.emb_dim, word_dim=args.word_dim, vocab_file=args.vocab_file, vocab_size=args.vocab_size)
    if args.model=='stt':
        image_embeddings_t, text_embeddings_t, im_pred_words, sent_pred_words  = model.build_stt_model(image, caption, None, seq_len, None, args)
    else:
        raise ValueError('Invalid Model')
    
    if args.use_abs:
        image_embeddings_t = tf.abs(image_embeddings_t)
        text_embeddings_t = tf.abs(text_embeddings_t)

    # Define the arrays for embedding vectors
    if args.num is None: 
        raise ValueError('Invalid configuration !!')
    image_embeddings_val=np.zeros((args.num, args.emb_dim))
    text_embeddings_val=np.zeros((args.num, args.emb_dim))

    print "Total number of validation samples: {}".format(args.num)

    # Define a saver
    saver=tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    i=0
    # image_captions=open('/home/dp1248/cvs/show-translate-tell/captioning/new_captions/ic_stt_coco_1024_150001.txt', 'w')
    with tf.Session(config=config) as sess:
        sess.run(tf.tables_initializer())
        saver.restore(sess, args.checkpoint)
        start_time = time.time()
        for i in range(0, args.num, args.batch_size):
            if i%5000==0: print "Processed: {}".format(i)
            try:
                ie, te, iw, sw, cap = sess.run([image_embeddings_t, text_embeddings_t, im_pred_words, sent_pred_words, caption])
                # sw=sw.T
                # for k in range(sw.shape[0]):
                    # curr_caption = list(sw[k, :])
                    # actual_caption = []
                    # for ele in curr_caption:
                        # if ele not in ['<unk>', '</s>', '<s>']:
                            # actual_caption.append(ele)
                            
                    # image_captions.write(' '.join(actual_caption)+ '\n')              
                image_embeddings_val[i:i+args.batch_size, :] = np.squeeze(ie)
                text_embeddings_val[i:i+args.batch_size, :] = np.squeeze(te)

            except tf.errors.OutOfRangeError:
                break
    # image_captions.close()
    print "Time taken for processing: {}".format(time.time()-start_time)
    # Average over 5 folds
    results=[]
    for fold in range(args.num_folds):
        print 'Fold: {}'.format(fold)
        ri, ri0, i2t_ranked_idx = i2t_gpu(image_embeddings_val[5000*fold: 5000*fold + 5000], text_embeddings_val[5000*fold: 5000*fold + 5000], measure=args.measure)
        print "Image to Text: "
        print "R@1: {} R@5: {} R@10 : {} Med: {} Mean: {}".format(ri[0], ri[1], ri[2], ri[3], ri[4])
        rt, rt0, t2i_ranked_idx = t2i_gpu(image_embeddings_val[5000*fold: 5000*fold + 5000], text_embeddings_val[5000*fold: 5000*fold + 5000], measure=args.measure)
        print "Text to Image: "
        print "R@1: {} R@5: {} R@10 : {} Med: {} Mean: {}".format(rt[0], rt[1], rt[2], rt[3], rt[4])
        print '---------------------------------------------'
        results += [list(ri) + list(rt)]
    
    print("Mean metrics: ")
    mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
          mean_metrics[:5])
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
          mean_metrics[5:10])
    # pdb.set_trace()
    if args.retrieve_text:
        # pdb.set_trace()
        test_file = open(args.val_ids_path, 'r').readlines()
        test_caps_file = open(args.val_caps_path, 'r').readlines()
        test_captions = [cap.strip() for cap in test_caps_file]
        test_images = [ele.strip() for ele in test_file]
        sample = args.test_sample
        sample_idx = test_images.index(sample)
        all_retrieved_idx = i2t_ranked_idx[sample_idx]
        top_3_idx = all_retrieved_idx[:3]
        retrieved_caps = []
        print "Top 3 captions: "
        for idx in top_3_idx:
            print test_captions[idx] 
        print "------------------------------------"
        print "GT captions: "
        for idx in range(5*sample_idx, 5*sample_idx+5): 
            print test_captions[idx]

    elif args.retrieve_image:
        test_file = open(args.val_ids_path, 'r').readlines()
        test_caps_file = open(args.val_caps_path, 'r').readlines()
        test_captions = [cap.strip() for cap in test_caps_file]
        test_images = [ele.strip() for ele in test_file]
        sample = args.test_sample
        sample_idx = test_captions.index(sample)
        all_retrieved_idx = t2i_ranked_idx[sample_idx]
        top_3_idx = all_retrieved_idx[:3]
        retrieved_caps = []
        print "Top 3 Images: "
        for idx in top_3_idx:
            print test_images[idx] 
	

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--num', type=int, default=None, help="Number of examples to be evaluated")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs to be evaluated")
    parser.add_argument('--emb_dim', type=int, default=1024, help="Batch size")
    parser.add_argument('--word_dim', type=int, default=300, help="Word Embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--margin', type=float, default=0.05, help="Margin for sim loss")
    parser.add_argument('--precompute', action='store_true', help="Flag to use precomputed CNN features")
    parser.add_argument('--use_l2_norm', action='store_true', help="Flag to use precomputed CNN features")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds for Cross validation")
    parser.add_argument('--vocab_size', type=int, default=26735, help="Size of the vocabulary")
    parser.add_argument('--num_units', type=int, default=1024, help="Number of hidden RNN units")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in RNN network")
    parser.add_argument('--test_sample', type=str, default='COCO_val2014_000000483108.jpg', help="Test captions path")
    parser.add_argument('--vocab_file', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc', help="Val file")
    parser.add_argument('--val_ids_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/test.ids', help="Test IDs path")
    parser.add_argument('--val_caps_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/test_caps.txt', help="Test captions path")
    parser.add_argument('--measure', type=str, default='cosine', help="Type of measure")
    parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/coco_val_precompute.tfrecord', help="Path to val tfrecord")
    parser.add_argument('--root_path', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/val2014', help="Experiment dir")
    parser.add_argument('--checkpoint', type=str, default='/shared/kgcoe-research/mil/peri/flowers_data/checkpoints_CMR_finetune_2018-08-11_16_45/model.ckpt-28000', help="LSTM checkpoint")
    parser.add_argument('--model', type=str, default='vse', help="Name of the model")
    parser.add_argument('--mode', type=str, default='val', help="Training or validation")
    parser.add_argument('--base', type=str, default='resnet_v2_152', help="Base architecture")
    parser.add_argument('--use_abs', action='store_true', help="use_absolute values for embeddings")
    parser.add_argument('--finetune_with_cnn', action='store_true', help="Finetune with CNN or not")
    parser.add_argument('--retrieve_text', action='store_true', help="retrieve_text or not")
    parser.add_argument('--retrieve_image', action='store_true', help="retrieve_image or not")
    args = parser.parse_args()
    print '--------------------------------'
    for key, value in vars(args).items():
        print key, ' : ', value
    print '--------------------------------'
    eval(args)