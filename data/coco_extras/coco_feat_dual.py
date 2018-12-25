import tensorflow as tf
import numpy as np
import pdb
import argparse
import itertools
import os

def preprocess_caption(caption):
    """
    Preprocess caption
    """
    new_caption = caption.replace('.', '').replace('\'', '').lower()
    new_caption = new_caption.replace(',', '').strip()
    
    return new_caption
    
def _bytes_feature(value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        
def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
    
def write_tfrecord(feature, caption):
    feature_lists = tf.train.FeatureLists(feature_list={"caption": _bytes_feature_list(caption.split(' '))})
    context = tf.train.Features(feature={"image": _bytes_feature(feature.tostring())})
                                
    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists) #
    
    return sequence_example

def main(args):
    # features = np.load(os.path.join(args.root_path, args.phase+'_ims.npy'))
    features = np.load('/shared/kgcoe-research/mil/peri/scan_data/data/f30k_precomp/test_ims.npy').astype(np.float32)[0:5000:5, :, :]
    captions = open(os.path.join(args.root_path, 'test_caps.txt'), 'r').readlines()
    num_samples = features.shape[0]
    tfrecord_writer=tf.python_io.TFRecordWriter(os.path.join(args.record_path, args.phase+'.tfrecord'))
    print "Total number of samples: {}".format(num_samples)
    for i in range(num_samples):
        if i%1000==0: print "Processed: {}".format(i)
        image_feature = features[i]
        cap=[]
        for j in range(i*5, i*5 + 5):
            sequence_example= write_tfrecord(image_feature, preprocess_caption(captions[j]))
            tfrecord_writer.write(sequence_example.SerializeToString())
            
    tfrecord_writer.close()
    print "Done writing TFrecord"
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/shared/kgcoe-research/mil/peri/scan_data/data/f30k_precomp', help='Path to features')
    parser.add_argument('--record_path', default='/shared/kgcoe-research/mil/cvs_cvpr18/coco', help='Path to features')
    parser.add_argument('--phase', default='f30k_scan_test_r152', help='Training or testing')
    args=parser.parse_args()
    main(args)