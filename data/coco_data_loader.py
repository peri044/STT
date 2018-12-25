import tensorflow as tf
import numpy as np
import os
from preprocessing import preprocessing_factory
import argparse
import pdb

class CocoDataLoader(object):
    """
    Data loader and writer object for MSCOCO dataset
    """
    def __init__(self, path=None, precompute=False, use_random_crop=False, model='stt'):
        self.data_path=path
        self.precompute=precompute
        self.use_random_crop=use_random_crop
        self.model=model
        
    def _int64_feature(self, value):
        """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
    def _bytes_feature(self, value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        
    def _bytes_feature_list(self, values):
        """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
        return tf.train.FeatureList(feature=[self._bytes_feature(v) for v in values])
        
    def _process_caption(self, caption):
        
        processed_caption = caption.replace(',', '').replace('\'', '').lower()
        processed_caption = processed_caption.replace('.', '')
        return processed_caption
        
    def _make_single_example(self, image_path, encoder_caption, decoder_caption, precompute=False):
        """
        Make a single example in a TF record
        """
        #Pre-process and split the encoder and decoder captions
        encoder_caption=self._process_caption(encoder_caption)
        encoder_caption_list = encoder_caption.split(' ')
        decoder_caption=self._process_caption(decoder_caption)
        decoder_caption_list = decoder_caption.split(' ')
        if not precompute:
            image = tf.gfile.FastGFile(image_path, "rb").read()
            
            feature_lists = tf.train.FeatureLists(feature_list={"encoder_caption": self._bytes_feature_list(encoder_caption_list),
                                                                "decoder_caption": self._bytes_feature_list(decoder_caption_list)})
            context = tf.train.Features(feature={
                            "image": self._bytes_feature(image)})
                            
        else:
            feature_lists = tf.train.FeatureLists(feature_list={"encoder_caption": self._bytes_feature_list(encoder_caption_list),
                                                                "decoder_caption": self._bytes_feature_list(decoder_caption_list)})
            context = tf.train.Features(feature={"image": self._bytes_feature(image_path.tostring())})
                                        
        sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

        return sequence_example
        
    def _make_dataset(self, phase, record_path, num=None):
        """
        Write the whole dataset to a TF record.
        """
        enc_caps = open(os.path.join(args.data_path, phase+'_enc.txt')).readlines()
        dec_caps = open(os.path.join(args.data_path, phase+'_dec.txt')).readlines()
        img_ids = open(os.path.join(args.data_path, phase+'_img_ids.txt')).readlines()
        tfrecord_writer = tf.python_io.TFRecordWriter(record_path)

        if num is None:
            num=len(img_ids)
        count=0

        for im_idx in range(num):
            if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num)
            if img_ids[im_idx].strip().find('train2014') !=-1:
                image = os.path.join(args.train_dir, img_ids[im_idx].strip())
            elif img_ids[im_idx].strip().find('val2014') !=-1:
                image = os.path.join(args.val_dir, img_ids[im_idx].strip())
            else:
                raise ValueError("Invalid Image")
                
            example = self._make_single_example(image, enc_caps[im_idx].strip(), dec_caps[im_idx].strip())
            tfrecord_writer.write(example.SerializeToString())
            count+=1
            
        print "Done generating TF records"
        
    def _precomputed_dataset(self, phase, record_path, feature_path, captions_path, num=None):
        """
        Write the whole dataset to a TF record.
        """
        
        enc_caps = open(os.path.join(args.captions_path, 'enc_2.txt')).readlines()
        dec_caps = open(os.path.join(args.captions_path, 'testall_caps.txt')).readlines()
        features=np.load(feature_path).astype(np.float32)[0:25000:5, :, :]
        tfrecord_writer = tf.python_io.TFRecordWriter(record_path)

        if num is None:
            num=len(features)
        count=0
        for im_idx in range(num):
            if count%1000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)                
            for cap_idx in range(im_idx*5, im_idx*5 +5):
                example = self._make_single_example(features[im_idx], enc_caps[cap_idx].strip(), dec_caps[cap_idx].strip(), precompute=True)
                tfrecord_writer.write(example.SerializeToString())
                count+=1
            
        print "Done generating TF records"
        
    def _inception_preprocess(self, image):
        
        """
        Pre-processing for inception. Convert the range to [-1, 1]
        """
        return (2.0/255)*image -1.
        
    def _vgg_preprocess(self, image):
        
        """
        Pre-processing for VGG.
        """
        image_preprocessing_fn = preprocessing_factory.get_preprocessing('resnet_v1_152', is_training=self.use_random_crop)
        return image_preprocessing_fn(image, 224, 224)
        
    def _parse_single_example(self, example_proto):
        context, sequence = tf.parse_single_sequence_example(
                                        example_proto,
                                        context_features={
                                          "image": tf.FixedLenFeature([], dtype=tf.string),
                                        },
                                        sequence_features={
                                          "encoder_caption": tf.FixedLenSequenceFeature([], dtype=tf.string),
                                          "decoder_caption": tf.FixedLenSequenceFeature([], dtype=tf.string)
                                        })
        if not self.precompute:
            image = tf.image.decode_jpeg(context["image"], channels=3)
            image = self._vgg_preprocess(image)
        else:
            image = tf.decode_raw(context['image'], out_type=tf.float32)
            if self.model=='stt':
                image_shape=[2048]
            elif self.model=='stt-att':
                image_shape=[36, 2048]
            elif self.model=='stt-para-att':
                image_shape=[36, 2048]
            image = tf.reshape(image, image_shape)
            
        encoder_caption = tf.cast(sequence["encoder_caption"], tf.string)[:50] # max_len allowed is 50
        decoder_caption = tf.cast(sequence["decoder_caption"], tf.string)[:50] # max_len allowed is 50
 
        return image, encoder_caption, decoder_caption, tf.size(encoder_caption), tf.size(decoder_caption)
        
    def _parse_val_example(self, example_proto):
        context, sequence = tf.parse_single_sequence_example(
                                        example_proto,
                                        context_features={
                                          "image": tf.FixedLenFeature([], dtype=tf.string),
                                        },
                                        sequence_features={
                                          "caption": tf.FixedLenSequenceFeature([], dtype=tf.string)
                                        })
        if not self.precompute:
            image = tf.image.decode_jpeg(context["image"], channels=3)
            image = self._vgg_preprocess(image)
        else:
            image = tf.decode_raw(context['image'], out_type=tf.float32)
            if self.model=='stt':
                    image_shape=[2048]
            elif self.model=='stt-att':
                image_shape=[36, 2048]
            elif self.model=='stt-para-att':
                image_shape=[36, 2048]
                
            image = tf.reshape(image, image_shape)
            
        caption = tf.cast(sequence["caption"], tf.string)[:50] # max_len allowed is 50
        # rev_caption = tf.reverse(caption, axis=[0])
        return image, caption, tf.size(caption)

    def _read_data(self, record_path, batch_size, phase='train', num_epochs=10):
        """
        Read the TF record and return the data
        """
        if phase !='val':
            dataset = tf.data.TFRecordDataset(record_path)
            
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.map(map_func=self._parse_single_example, num_parallel_calls=4)
            # Add in sequence lengths.
            dataset = dataset.map(lambda im, enc_c, dec_c, enc_len, dec_len: (im, enc_c, tf.concat([['<s>'], dec_c], axis=0), \
                                                                                tf.concat([dec_c, ['</s>']], axis=0), enc_len, dec_len+1), \
                                                                                num_parallel_calls=4).prefetch(buffer_size=batch_size)
            # Pads the captions to the max caption size in a batch
            if not self.precompute:
                image_shape=[224, 224, 3]
            else:
                if self.model=='stt':
                    image_shape=[2048]
                elif self.model=='stt-att':
                    image_shape=[36, 2048]
                elif self.model=='stt-para-att':
                    image_shape=[36, 2048]
            dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, 
                                           padded_shapes=(
                                                image_shape,  # img
                                                tf.TensorShape([None]),  # enc_caption
                                                tf.TensorShape([None]),  # dec_caption
                                                tf.TensorShape([None]),  # dec_caption
                                                tf.TensorShape([]),# src_len
                                                tf.TensorShape([])),# targ_len
                                           padding_values=(0.,  # src 
                                                          '</s>',  # enc_caption_pad
                                                          '</s>',  # dec_caption_pad
                                                          '</s>',  # dec_caption_pad
                                                           0,  # Enc Seq len
                                                           0))) # Dec Seq len
        
            dataset = dataset.repeat(num_epochs)
            iterator = dataset.make_one_shot_iterator()
            image, encoder_caption, decoder_input_caption, decoder_output_caption, enc_len, dec_len = iterator.get_next()
            
            return image, encoder_caption, decoder_input_caption, decoder_output_caption, enc_len, dec_len
           
        else: 
            dataset = tf.data.TFRecordDataset(record_path)
            dataset = dataset.map(map_func=self._parse_val_example, num_parallel_calls=4)
            if not self.precompute:
                image_shape=[224, 224, 3]
            else:
                if self.model=='stt':
                    image_shape=[2048]
                elif self.model=='stt-att':
                    image_shape=[36, 2048]
                
            dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, 
                               padded_shapes=(
                                    image_shape,  # img
                                    tf.TensorShape([None]),  # caption
                                    tf.TensorShape([])),# src_len
                               padding_values=(0.,  # src 
                                              '</s>',  # caption_pad
                                               0))) # Seq len
            
            dataset = dataset.repeat(num_epochs)
            iterator = dataset.make_one_shot_iterator()
            image, caption, seq_len = iterator.get_next()

            return image, caption, seq_len
            
    def _read_para_data(self, record_path, batch_size, num_epochs=10):
        """
        Read the TF record and return the data
        """
        dataset = tf.data.TFRecordDataset(record_path)
        dataset = dataset.map(map_func=self._parse_single_example, num_parallel_calls=4)

        # Pads the captions to the max caption size in a batch
        if not self.precompute:
            image_shape=[224, 224, 3]
        else:
            if self.model=='stt':
                image_shape=[2048]
            elif self.model=='stt-att':
                image_shape=[36, 2048]
            elif self.model=='stt-para-att':
                image_shape=[36, 2048]
        dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, 
                                       padded_shapes=(
                                            image_shape,  # img
                                            tf.TensorShape([None]),  # enc_caption
                                            tf.TensorShape([None]),  # dec_caption
                                            tf.TensorShape([]),# src_len
                                            tf.TensorShape([])),# targ_len
                                       padding_values=(0.,  # src 
                                                      '</s>',  # enc_caption_pad
                                                      '</s>',  # dec_caption_pad
                                                       0,  # Enc Seq len
                                                       0))) # Dec Seq len
    
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        image, encoder_caption, decoder_caption, enc_len, dec_len = iterator.get_next()
        
        return image, encoder_caption, decoder_caption, enc_len, dec_len

def main(args):

    dataset = CocoDataLoader(args.data_path)
    # Make the dataset
    if not args.precompute:
        dataset._make_dataset(args.phase, args.record_path, num=args.num)
    else:
        dataset._precomputed_dataset(args.phase, args.record_path, args.feature_path, args.captions_path, num=args.num)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/stt_data')
    parser.add_argument('--train_dir', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/train2014')
    parser.add_argument('--val_dir', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/val2014')
    parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/coco_stt_train.tfrecord')
    parser.add_argument('--feature_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/karpathy/oe_coco/data/coco/images/10crop/train.npy')
    parser.add_argument('--captions_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/karpathy/oe_coco/data/coco')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--precompute', action='store_true', help='Flag to build using precomputed CNN features')
    parser.add_argument('--num', type=int, default=None)
    args = parser.parse_args()
    main(args)