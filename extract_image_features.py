import tensorflow as tf
import numpy as np
import argparse
from dnn_library import *
from skimage import io
from skimage.transform import resize
from preprocessing import preprocessing_factory
from nets import *
import os
import cv2
import pdb

def vgg_preprocess(image, base_arch):
    
    """
    Pre-processing for base network.
    """
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(base_arch, is_training=True)
    return tf.expand_dims(image_preprocessing_fn(image, 224, 224), 0)
    
def preprocess_caption(caption):
    """
    Preprocess caption
    """
    new_caption = caption.replace('.', '')
    new_caption = new_caption.replace(',', '').strip()
    
    return new_caption
    
def load_flickr_data(args):
    """
    Load flickr data
    """
    images_path = open(os.path.join(args.data_path, 'train.token')).readlines()
    image_data=[]
    caption_data=[]
    for sample in images_path:
        sample=sample.strip()
        image=sample.split('#')[0]
        caption = sample.split('\t')[1]
        preprocessed_caption = preprocess_caption(caption)
        image_data.append(os.path.join(args.root_path, image))
        caption_data.append(preprocessed_caption)
        
    print "Done loading data"
    return image_data, caption_data
    
def load_coco_data(args):
	"""
	Load MSCOCO data
	"""
	enc_caps = open(os.path.join(args.data_path, args.phase+'_enc.txt')).readlines()
	dec_caps = open(os.path.join(args.data_path, args.phase+'_dec.txt')).readlines()
	img_ids = open(os.path.join(args.data_path, args.phase+'_img_ids.txt')).readlines()
		
	return img_ids, enc_caps, dec_caps
    
def _bytes_feature(value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        
def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
    
def write_tfrecord(feature, enc_caption, dec_caption):
    encoder_caption_list = enc_caption.split(' ')
    decoder_caption_list = dec_caption.split(' ')
    feature_lists = tf.train.FeatureLists(feature_list={"encoder_caption": _bytes_feature_list(encoder_caption_list),
                                                        "decoder_caption": _bytes_feature_list(decoder_caption_list)})
    context = tf.train.Features(feature={"image": _bytes_feature(feature.tostring())})
                                
    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    
    return sequence_example
       
        
def feature_extractor(args, image, reuse=None, is_training=False):
		"""
		Builds the model architecture
		"""			
		# Define the network and pass the input image
		with slim.arg_scope(model[args.base_arch]['scope']):
				logits, end_points = model[args.base_arch]['net'](image, num_classes=1001, is_training=is_training) #model[args.base_arch]['num_classes']
			
		# features 
		feat_anchor = tf.squeeze(end_points[model[args.base_arch]['end_point']])
			
		return feat_anchor
        
def main(args):
    
    # Define the input 
    input_image = tf.placeholder(shape=[None, None, 3], dtype=tf.float32, name='input_image')
    preprocessed_image = vgg_preprocess(input_image, args.base_arch)
    
    # Extract the features
    features = feature_extractor(args, preprocessed_image, is_training=False)
    
    # Define the saver
    saver = tf.train.Saver()
    
    # Load the data file
    if args.dataset=='flickr':
        image_data, caption_data = load_flickr_data(args)
        interval=5
    elif args.dataset=='mscoco':
        image_data, enc_caps, dec_caps = load_coco_data(args)
        interval=1
        
	print "Total number of samples: {}".format(len(image_data))
	image_features = np.zeros((len(image_data), features.shape.as_list()[0]))

	# Define the TF record writer
	tfrecord_writer = tf.python_io.TFRecordWriter(args.record_path)
	count=0
	image_to_feature={}
    with tf.Session() as sess:
        # Restore pre-trained weights
        saver.restore(sess, args.checkpoint)
        # try:
        for im_idx in range(len(image_data)):
            if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, len(image_data))
            if image_data[im_idx].strip().find('train2014') !=-1:
                image_path = os.path.join(args.train_dir, image_data[im_idx].strip())
            elif image_data[im_idx].strip().find('val2014') !=-1:
                image_path = os.path.join(args.val_dir, image_data[im_idx].strip())
            else:
                raise ValueError("Invalid Image")

            # sample is of form (image, caption)
            image = io.imread(image_path)
            if len(image.shape)!=3:
                # print "Found Gray-scale image at : {}".format(im_idx)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Run  the session to extract features
            feature_val = sess.run(features, feed_dict={input_image: image})

            sequence_example = write_tfrecord(feature_val, preprocess_caption(enc_caps[im_idx]), preprocess_caption(dec_caps[im_idx]))
            tfrecord_writer.write(sequence_example.SerializeToString())

            image_features[count] = feature_val
            count+=1
        # except:
            # pdb.set_trace()
            # print "done"
        tfrecord_writer.close()
        np.save(os.path.join(args.data_path, 'coco_train_pre_dual_'+args.base_arch+'.npy'), image_features)
        print "Total number of image features: {}".format(count)
        print "Done extracting Image features !!"
       
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mscoco', help='Data file')
    parser.add_argument('--phase', type=str, default='train', help='phase')
    parser.add_argument('--data_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/stt_data', help='Data file')
    parser.add_argument('--train_dir', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/train2014', help='Data file')
    parser.add_argument('--val2014', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/val2014', help='Data file')
    parser.add_argument('--root_path', type=str, default='/shared/kgcoe-research/mil/Flickr30k/flickr30k_images/flickr30k_images', help='Root_path')
    parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/stt_data/train_dual_precomputed_r152.tfrecord', help='Root_path')
    parser.add_argument('--base_arch', type=str, default='resnet_v2_152', help='Data file')
    parser.add_argument('--checkpoint', type=str, default='/shared/kgcoe-research/mil/peri/tf_checkpoints/resnet_v2_152.ckpt', help='Path to checkpoint')
    args=parser.parse_args()
    main(args)
