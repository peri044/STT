import tensorflow as tf
import numpy as np
from dnn_library import *
from nets import *
import pdb
from attention import *
from text_encoder import *

class STT(object):
    """
    Base class for Cross-Modal Retrieval experiments
    """
    def __init__(self, base='inception_v1', vocab_file='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc', margin=1., embedding_dim=512,word_dim=1024, vocab_size=26735):
        self.base_arch = base
        self.margin=margin
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_dim = word_dim

    def process_text(self):
        """
        Loads the vocabulary and builds the vocab to idx tables.
        Builds the embedding matrix given the vocab and hidden dimension size
        """
        self.vocab_table = tf.contrib.lookup.index_table_from_file(self.vocab_file, default_value=0)
        self.reverse_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(self.vocab_file)
        self.embedding_matrix = tf.get_variable("embeddings/embedding_share", shape=[self.vocab_size, self.word_dim], \
                                                                                    trainable=True, \
                                                                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1), \
                                                                                    dtype=tf.float32)
        
    def _feature_extractor(self, image, reuse=None, is_training=True):
        """
        Builds the model architecture
        """			
        # Define the network and pass the input image
        with tf.variable_scope('Feature_extractor', reuse=reuse) as scope:
            with slim.arg_scope(model[self.base_arch]['scope']):
                logits, end_points = model[self.base_arch]['net'](image, num_classes=model[self.base_arch]['num_classes'], is_training=is_training)

        # Avg pool features of inception v1 (size: 1024)
        feat_anchor = end_points[model[self.base_arch]['end_point']]  ## Dropout_0b
        feat_anchor = tf.squeeze(end_points[model[self.base_arch]['end_point']])
            
        return feat_anchor
        
    def build_rnn_cell(self, num_units, num_layers, dropout=0.):
        """
        Define the RNN cell with num_units, num_layers and dropout (Dropout=0 for inference)
        """
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.GRUCell(num_units, \
                                                                            kernel_initializer=tf.orthogonal_initializer(),
                                                                            reuse=tf.AUTO_REUSE), \
                                                                            input_keep_prob=(1.0 - dropout)) for _ in range(num_layers)])
        
        return cell
    
    def _projection_layer(self):
		"""
		Builds the projection layer
		"""
		# Get the projection layer
		with tf.variable_scope("build_network"):
			with tf.variable_scope("decoder/output_projection"):
				projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False, name="output_projection")
				
		return projection_layer
        
    def _build_encoder(self, input_seq, num_units, num_layers, seq_len, dropout=0.):
        """
        Builds the encoder part of seq-seq model
        """
        # Look up the ids for input sequence
        input_word_ids = self.vocab_table.lookup(input_seq)
        # Time major
        input_word_ids = tf.transpose(input_word_ids)
        
        # Define the encoder cell and build input embeddings
        self.cell = self.build_rnn_cell(num_units, num_layers, dropout=dropout)
        encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, input_word_ids)
        with tf.variable_scope("dynamic_seq2seq", dtype=tf.float32):
            with tf.variable_scope("encoder") as scope:
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                                                    self.cell,
                                                    encoder_emb_inp,
                                                    dtype=tf.float32,
                                                    sequence_length=seq_len,
                                                    time_major=True,
                                                    swap_memory=True)
                                                
        return encoder_outputs, encoder_state
        
    def _build_decoder(self, decoder_initial_state, decoder_input, dec_len, params, phase='train', reuse=tf.AUTO_REUSE, cell=None):
        """
        Builds the decoder part of seq-seq model
        """
        with tf.variable_scope("dynamic_seq2seq", dtype=tf.float32, reuse=reuse):
            with tf.variable_scope("decoder") as decoder_scope:
                # Define RNN cell
                if cell is not None:
                    cell=cell
                else:
                    cell = self.build_rnn_cell(params.num_units, params.num_layers, dropout=params.dropout)	
                    
                if phase=='train':
                    # Look up the ids for input sequence
                    decoder_word_ids = self.vocab_table.lookup(decoder_input)
                    # Time major
                    decoder_word_ids = tf.transpose(decoder_word_ids)
                    # Look up embedding vectors for decoder inputs
                    decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, decoder_word_ids)
                    # Helper
                    helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, dec_len, time_major=True)

                    # Decoder
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)

                    # Dynamic decoding
                    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major=True, swap_memory=True, scope=decoder_scope)
                    
                    # Get the logits from the projection layer
                    projection_layer = self._projection_layer()
                    logits = projection_layer(outputs.rnn_output)
                    
                    sample_id = outputs.sample_id
                    
                elif phase=='inference':
                    # Look up for start and end tokens
                    start_token_id = tf.cast(self.vocab_table.lookup(tf.constant('<s>')), tf.int32)
                    end_token = tf.cast(self.vocab_table.lookup(tf.constant('</s>')), tf.int32)
                    start_tokens = tf.fill([100], start_token_id)

                    # Greedy decoding
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_matrix, start_tokens, end_token)

                    # Decoder
                    projection_layer=self._projection_layer()
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state, output_layer=projection_layer)

                    # Dynamic decoding
                    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                        maximum_iterations=20,
                                                        output_time_major=True,
                                                        swap_memory=True,
                                                        scope=decoder_scope)
                                                        
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

            return logits, sample_id, final_context_state
        
    def _build_embedding(self, feat_anchor, embedding_dim=512, scope_name="embedding", act_fn=None, reuse=None):
        """
        Build the embedding network
        """
        with slim.arg_scope([slim.fully_connected],
                             activation_fn=act_fn,
                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                             weights_regularizer=slim.l2_regularizer(0.0002),
                             reuse=reuse): 
            embedding = slim.fully_connected(feat_anchor, embedding_dim, activation_fn=act_fn, scope=scope_name)

        return embedding
       
    def build_stt_model(self, images, encoder_captions, decoder_captions, enc_len, dec_len, params, reuse=None):
        """
        Builds the Show, Translate and Tell model
        """
        # Build the embeddings for images.
        if not params.precompute:
            image_features = self._feature_extractor(images, is_training=params.finetune_with_cnn, reuse=reuse)
        else: 
            image_features = images
        # Build the embeddings for images, encoder_captions
        image_embeddings = self._build_embedding(image_features, self.embedding_dim, act_fn=None, reuse=reuse, scope_name='image_embedding')
        
        # Build the encoder and its embeddings
        self.process_text()
        encoder_outputs, encoder_state = self._build_encoder(encoder_captions, params.num_units, params.num_layers, enc_len, dropout=params.dropout)
        encoder_features = tf.concat(encoder_state, axis=1, name='encoder_features')
        text_embeddings = self._build_embedding(encoder_features, self.embedding_dim, act_fn=None, reuse=reuse, scope_name='text_embedding')
        
        # L2 normalize the embeddings
        image_embeddings = tf.nn.l2_normalize(image_embeddings, axis=1, name='norm_image_embeddings')
        text_embeddings = tf.nn.l2_normalize(text_embeddings, axis=1, name='norm_text_embeddings')
            
        dec_image_logits, dec_sent_logits= None, None
        if params.mode =='val': 
            phase = 'inference'
        else:
            phase='train'
        # Build the decoder with encoder text embeddings
        text_embed_split = tuple(tf.split(text_embeddings, num_or_size_splits=params.num_layers, axis=1))
        dec_sent_logits, dec_sent_sample_id, sent_context_state = self._build_decoder(text_embed_split, decoder_captions, dec_len, params, phase=phase)
        
        # Build the decoder with image embeddings
        image_embed_split = tuple(tf.split(image_embeddings, num_or_size_splits=params.num_layers, axis=1))
        dec_image_logits, dec_im_sample_id, image_context_state = self._build_decoder(image_embed_split, decoder_captions, dec_len, params, phase=phase, reuse=True)
        if params.mode=='val':
            im_pred_words = self.reverse_vocab_table.lookup(tf.cast(dec_im_sample_id, tf.int64))
            sent_pred_words = self.reverse_vocab_table.lookup(tf.cast(dec_sent_sample_id, tf.int64))
            return image_embeddings, text_embeddings, im_pred_words, sent_pred_words
            
        return image_embeddings, text_embeddings, dec_image_logits, dec_sent_logits
        
    def build_stt_attention_model(self, images, encoder_captions, decoder_captions, enc_len, dec_len, params, reuse=None):
        """
        Builds the Show, Translate and Tell model
        """
        # Build the embeddings for images.
        if not params.precompute:
            image_features = self._feature_extractor(images, is_training=params.finetune_with_cnn, reuse=reuse)
        else: 
            image_features = images

        # Build the embeddings for images, encoder_captions
        image_embeddings = self._build_embedding(image_features, self.embedding_dim, act_fn=None, reuse=reuse, scope_name='image_embedding')
        # Avg pool the regions
        mean_image_embeddings = tf.reduce_mean(image_embeddings, axis=1)
        mean_norm_image_embeddings = tf.nn.l2_normalize(mean_image_embeddings, axis=1)
        image_embeddings = tf.nn.l2_normalize(image_embeddings, axis=2, name='norm_image_reg_embeddings')

        #Build the embeddings for captions
        self.process_text()
        # Look up the ids for input sequence
        input_word_ids = self.vocab_table.lookup(encoder_captions)
        encoder_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, input_word_ids)
        scan = SCAN()
        word_features, text_features = scan._build_text_encoder(encoder_embeddings, params, enc_len)
        word_features = tf.nn.l2_normalize(word_features, axis=2)
        sim_matrix = t2i_attention(image_embeddings, word_features, enc_len, params)

        text_features = tf.nn.l2_normalize(text_features, axis=1)
        
        input_word_ids = self.vocab_table.lookup(encoder_captions)
        encoder_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, input_word_ids)
        scan = SCAN()
        word_features, text_features = scan._build_text_encoder(encoder_embeddings, params, enc_len)
        
        dec_image_logits, dec_sent_logits= None, None
        if params.mode =='val': 
            phase = 'inference'
        else:
            phase='train'
        # Build the decoder with encoder text embeddings
        text_embed_split = tuple(tf.split(text_features, num_or_size_splits=params.num_layers, axis=1))
        dec_sent_logits, dec_sent_sample_id, sent_context_state = self._build_decoder(text_embed_split, decoder_captions, dec_len, params, phase=phase)
        
        # Build the decoder with image embeddings
        image_embed_split = tuple(tf.split(mean_norm_image_embeddings, num_or_size_splits=params.num_layers, axis=1))
        dec_image_logits, dec_im_sample_id, image_context_state = self._build_decoder(image_embed_split, decoder_captions, dec_len, params, phase=phase)
        if params.mode=='val':
            im_pred_words = self.reverse_vocab_table.lookup(tf.cast(dec_im_sample_id, tf.int64))
            sent_pred_words = self.reverse_vocab_table.lookup(tf.cast(dec_sent_sample_id, tf.int64))
            return image_embeddings, word_features, im_pred_words, sent_pred_words
            
        return mean_norm_image_embeddings, text_features, dec_image_logits, dec_sent_logits, sim_matrix
        
    def build_stt_att_t2t_model(self, images, encoder_captions, decoder_captions, enc_len, dec_len, params, reuse=None):
        """
        Builds the Show, Translate and Tell model
        """
        # Build the embeddings for images.
        if not params.precompute:
            image_features = self._feature_extractor(images, is_training=params.finetune_with_cnn, reuse=reuse)
        else: 
            image_features = images

        # Build the embeddings for images, encoder_captions
        image_embeddings = self._build_embedding(image_features, self.embedding_dim, act_fn=None, reuse=reuse, scope_name='image_embedding')
        # Avg pool the regions
        mean_image_embeddings = tf.reduce_mean(image_embeddings, axis=1)
        mean_norm_image_embeddings = tf.nn.l2_normalize(mean_image_embeddings, axis=1)
        image_embeddings = tf.nn.l2_normalize(image_embeddings, axis=2, name='norm_image_reg_embeddings')

        #Build the embeddings for captions
        self.process_text()
        # Look up the ids for input sequence
        input_word_ids = self.vocab_table.lookup(encoder_captions)
        encoder_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, input_word_ids)
        scan = SCAN()
        word_features, text_features = scan._build_text_encoder(encoder_embeddings, params, enc_len)
        word_features = tf.nn.l2_normalize(word_features, axis=2)
        sim_matrix = t2i_attention(image_embeddings, word_features, enc_len, params)

        text_features = tf.nn.l2_normalize(text_features, axis=1)

        # For paraphrases
        paraphrase_word_ids = self.vocab_table.lookup(decoder_captions)
        paraphrase_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, paraphrase_word_ids)
        para_word_features, para_text_features = scan._build_text_encoder(paraphrase_embeddings, params, dec_len)
        para_text_features = tf.nn.l2_normalize(para_text_features, axis=1)
        # para_word_features = tf.nn.l2_normalize(para_word_features, axis=2)
        # sim_para_matrix = t2i_attention(image_embeddings, para_word_features, dec_len, params)
        sim_para_matrix=None
        # Text to text similarity
        t2t_matrix = tf.matmul(text_features, para_text_features, transpose_b=True)
        
        dec_image_logits, dec_sent_logits= None, None
        if params.mode =='val': 
            phase = 'inference'
        else:
            phase='train'
        # Build the decoder with encoder text embeddings
        text_embed_split = tuple(tf.split(text_features, num_or_size_splits=params.num_layers, axis=1))
        dec_sent_logits, dec_sent_sample_id, sent_context_state = self._build_decoder(text_embed_split, decoder_captions, dec_len, params, phase=phase)
        
        # Build the decoder with image embeddings
        image_embed_split = tuple(tf.split(mean_norm_image_embeddings, num_or_size_splits=params.num_layers, axis=1))
        dec_image_logits, dec_im_sample_id, image_context_state = self._build_decoder(image_embed_split, decoder_captions, dec_len, params, phase=phase)
        if params.mode=='val':
            im_pred_words = self.reverse_vocab_table.lookup(tf.cast(dec_im_sample_id, tf.int64))
            sent_pred_words = self.reverse_vocab_table.lookup(tf.cast(dec_sent_sample_id, tf.int64))
            return image_embeddings, word_features, text_features, para_word_features, para_text_features, im_pred_words, sent_pred_words
            
        return mean_norm_image_embeddings, text_features, para_text_features, dec_image_logits, dec_sent_logits, sim_matrix, sim_para_matrix, t2t_matrix       
        
    def get_max_time(self, tensor):
        return tensor.shape[0].value or tf.shape(tensor)[0]
        
    def stt_loss(self, image_embeddings, text_embeddings, dec_image_logits, dec_sent_logits, decoder_target_caption, dec_len, args, sim_scores=None, sim_para_scores=None, t2t_scores=None):
        """
        Loss for Show, Translate and Tell model
        """
        # Look up the ids for target sequence
        output_word_ids = self.vocab_table.lookup(decoder_target_caption)
        # Time major
        output_word_ids = tf.transpose(output_word_ids)

        sim_loss, sent_loss, image_loss = self.sim_loss(image_embeddings, text_embeddings, args, sim_scores=sim_scores)
        t2t_loss=None
        sim_para_loss=None
        if sim_para_scores is not None:
            sim_para_loss, _, _ = self.sim_loss(image_embeddings, text_embeddings, args, sim_scores=sim_para_scores)
        if t2t_scores is not None:
            t2t_loss, _, _ = self.sim_loss(image_embeddings, text_embeddings, args, sim_scores=t2t_scores)
            
        image_captioning_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_word_ids, logits=dec_image_logits)
        seq2seq_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_word_ids, logits=dec_sent_logits)
        
        # Use target sequence mask to filter out outputs of padded words
        max_time = self.get_max_time(output_word_ids)
        target_weights = tf.sequence_mask(dec_len, max_time, dtype=dec_image_logits.dtype)
        target_weights = tf.transpose(target_weights)
        
        batch_size=image_embeddings.shape.as_list()[0]
        image_captioning_loss = tf.reduce_sum(image_captioning_ce * target_weights) / tf.to_float(batch_size)
        seq2seq_loss = tf.reduce_sum(seq2seq_ce * target_weights) / tf.to_float(batch_size)
   
        return seq2seq_loss, image_captioning_loss, sim_loss, sim_para_loss, t2t_loss
           
    def sim_loss(self, image_embeddings, text_embeddings, args, sim_scores=None):
        """
        Order violation or cosine similarity loss for image and text embeddings
        """

        with tf.name_scope('Sim_Loss') as scope:
            if sim_scores is None:
                if args.use_abs:
                    image_embeddings = tf.abs(image_embeddings)
                    text_embeddings = tf.abs(text_embeddings)
                    
                if args.measure=='cosine':
                    sim_scores = tf.matmul(image_embeddings, text_embeddings, transpose_b=True, name='sim_score')
                elif args.measure=='order':
                    # refer to eqn in paper or code of http://openaccess.thecvf.com/content_cvpr_2018/papers/Wehrmann_Bidirectional_Retrieval_Made_CVPR_2018_paper.pdf
                    im_emb = tf.expand_dims(image_embeddings, 0) # 1x128x2048
                    text_emb = tf.expand_dims(text_embeddings, 1) # 128x1x2048
                    im_emb = tf.tile(im_emb, [image_embeddings.shape.as_list()[0], 1, 1]) # 128x128x2048 (Each row has 128x2048 im_emb)
                    text_emb = tf.tile(text_emb, [1, text_embeddings.shape.as_list()[0], 1]) # 128x128x2048 (Each row has its text emb replicated 128 times)
                    sqr_diff = tf.square(tf.maximum(text_emb - im_emb, 0.)) 
                    sqr_diff_sum = tf.squeeze(tf.reduce_sum(sqr_diff, 2))
                    sim_scores = -tf.transpose(tf.sqrt(sqr_diff_sum), name='order_sim_scores') # Note the negative sign as this is a distance metric
                  
            # Get the diagonal of the matrix
            sim_diag = tf.expand_dims(tf.diag_part(sim_scores), 0, name='sim_diag')
            sim_diag_tile = tf.tile(sim_diag, multiples=[sim_diag.shape.as_list()[1], 1], name='sim_diag_tile')
            sim_diag_transpose = tf.transpose(sim_diag, name='sim_diag_transpose')
            sim_diag_tile_transpose = tf.tile(sim_diag_transpose, multiples=[1, sim_diag.shape.as_list()[1]], name='sim_diag_tile_transpose')

            # compare every diagonal score to scores in its column
            # caption retrieval
            loss_s = tf.maximum(args.margin + sim_scores - sim_diag_tile_transpose, 0.)
            # compare every diagonal score to scores in its row
            # image retrieval
            loss_im = tf.maximum(args.margin + sim_scores - sim_diag_tile, 0.)

            # clear the costs for diagonal elements
            mask = tf.eye(loss_s.shape.as_list()[0], dtype=tf.bool, name='Mask')
            mask_not = tf.cast(tf.logical_not(mask, name='mask_not'), tf.float32)
            
            neg_s_loss   = tf.multiply(loss_s, mask_not, name='neg_s_loss')
            neg_im_loss = tf.multiply(loss_im, mask_not, name='neg_im_loss')
            
            # Mining the hardest negative for each sample
            if args.mine_n_hard>0:
                if args.mine_n_hard==1:
                    loss_s = tf.reduce_max(neg_s_loss, axis=1)
                    loss_im = tf.reduce_max(neg_im_loss, axis=0)
                else:
                    loss_s = tf.contrib.framework.sort(neg_s_loss, axis=1, direction='DESCENDING')
                    loss_im = tf.contrib.framework.sort(neg_im_loss, axis=0, direction='DESCENDING')
                    # Build the index matrix to gather_nd
                    batch_size=loss_s.shape.as_list()[0]
                    indices= np.zeros((batch_size, mine_n_hard, 2))
                    for it in range(batch_size):
                        for m in range(mine_n_hard):
                            indices[it][m][0] = it
                            indices[it][m][1] = m

                    # Get the top N distances and reduce them
                    loss_s = tf.gather_nd(loss_s, indices.astype(np.int32))
                    loss_im = tf.gather_nd(loss_im, indices.astype(np.int32))
                
            loss_s = tf.reduce_sum(loss_s, name='loss_s')
            loss_im = tf.reduce_sum(loss_im, name='loss_im')
            total_loss = loss_s + loss_im

            return total_loss, loss_s, loss_im
            
            
			

		
		
		