import tensorflow as tf
import pdb

class Architectures(object):
    """
    Implementation of various text encoder architectures for cross-modal retrieval 
    """
    def __init__(self):
        pass
        
    def _gru_cell(self, params, scope_name):
        """
        Defines a GRU cell
        """
        gru_cell = tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.GRUCell(params.num_units, 
                                                                             name=scope_name, 
                                                                             reuse=tf.AUTO_REUSE), 
                                                                             input_keep_prob=(1.0 - params.dropout))
                                                                             
        return gru_cell
        
    def _dynamic_rnn(self, cell, inputs, seq_len):
        """
        Defines a dynamic RNN network which unrolls over inputs
        """
        _, output_state = tf.nn.dynamic_rnn(cell,
                                   inputs,
                                   dtype=tf.float32,
                                   sequence_length=seq_len, 
                                   time_major=False,
                                   swap_memory=False)
                                   
        return output_state
    
    def _build_hrne_encoder(self, seq_embeddings, seq_len, params, use_min_partition=False):
        """
        Builds hierarchical text encoder with specified stride
        """
        with tf.variable_scope('dynamic_seq2seq') as scope:
			#Define GRU cells for both the layers
			gru_1 = self._gru_cell(params, scope_name='gru_1')
			gru_2 = self._gru_cell(params, scope_name='gru_2')

			gru_1_len = params.stride
			gru_2_len = seq_embeddings.shape.as_list()[1]
			# Initialize the layer 1 and layer 2 hidden state.
			h_prev = tf.zeros([params.batch_size, params.num_units], name='h_init')

			output_gru_1=[]
			#Run the first layer of GRU and stack individual timesteps
			for step in range(gru_2_len):
				out, h_prev = gru_1(inputs=seq_embeddings[:,step,:], state=h_prev)
				output_gru_1.append(out)
			#Stack all the states and split the batch into individual samples
			stacked_states = tf.stack(output_gru_1, axis=1)
			state_dim=stacked_states.shape.as_list()[-1]
			batch_padded_state_vectors = tf.split(stacked_states, num_or_size_splits=params.batch_size, axis=0)
			batch_state_vectors=[]
			batch_strided_states=[]
			partitions=[]
			inter=[]
			for index in range(len(batch_padded_state_vectors)):
				# Get the vectors corresponding to the actual length of the sample caption
				sliced_state = tf.squeeze(tf.slice(batch_padded_state_vectors[index], begin=[0,0,0], size=[1, seq_len[index], state_dim]))
				final_timestep = sliced_state[-1, :]
				batch_state_vectors.append(final_timestep)
				# Get the strided outputs. Strided slice includes the first timestep as well. Ignore that !!
				strided_sliced_state = tf.strided_slice(sliced_state[:-1, :], [0,0], [seq_len[index], state_dim], [params.stride, 1])[1:, :]
				# Infer the partitions given by strided slice op
				inferred_partitions = tf.cast(tf.ceil(tf.divide(seq_len[index]-1, params.stride)), tf.int32) -1
				partitions.append(inferred_partitions)
				batch_strided_states.append(strided_sliced_state)
				inter.append(strided_sliced_state[-1, :])
			# Batch all the individual final timestep vectors back
			layer1_state_vectors = tf.stack(batch_state_vectors, axis=0)
			# Above looping caused dynamic shapes. 
			# Set the static shape to ensure rest of the graph builds with static shapes.
			layer1_state_vectors.set_shape([params.batch_size, state_dim])
			intermediate_state_vec = tf.stack(inter, axis=0)
			intermediate_state_vec.set_shape([params.batch_size, state_dim])
			# Use the minimum batch seq_len to determine partitions else consider the more general case
			layer2_input_states=[]
			if use_min_partition:
				min_batch_seq_len = tf.reduce_min(seq_len)
				minimum_partitions = tf.cast(tf.floor(tf.divide(tf.cast(min_batch_seq_len, tf.float32), params.stride)), tf.int32)
				# Slice out minimum partitions from strided states for each sample in the batch
				for strided_state in batch_strided_states:
					min_sliced_strided_state = tf.slice(strided_state, [0,0], [minimum_partitions, state_dim])
					layer2_input_states.append(min_sliced_strided_state)
					
				# Sequence length input to second layer should all be minimum partitions
				partitions = params.batch_size*[minimum_partitions]
			else:
				# Get the maximum length of sequences in a batch
				max_pad_len= tf.reduce_max(partitions)
				# Pad the rest of the samples to the maximum length sequence to form inputs to second GRU layer
				for k, state in enumerate(batch_strided_states):
					current_num_partitions = partitions[k]
					pad_value = max_pad_len - current_num_partitions
					constant_pad_vector = tf.pad(state, [[0, pad_value], [0, 0]])
					layer2_input_states.append(constant_pad_vector)
        
			# Stack all the batch minimum strided states
			stacked_layer2_input_states=tf.stack(layer2_input_states, axis=0, name='stacked_layer2_input_states')

			# Append the last state for comprehensive information
			all_layer2_input_states = tf.concat([stacked_layer2_input_states, tf.expand_dims(layer1_state_vectors, 1)], axis=1)

			# Form GRU_2 chain with all the strided states from layer 1
			# Gather the final state from GRU_2
			output_state = self._dynamic_rnn(gru_2, all_layer2_input_states, partitions+tf.ones_like(partitions)) # Since we are adding the final timestep vectors later

			# Concat outputs from both the layers
			final_concat_vector = tf.concat([layer1_state_vectors, output_state], axis=1)
                                 
        return final_concat_vector
        
    def _gated_fusion_unit(self, local_vector, global_vector):
        """
        Gated fusion unit to fuse local and global vectors
        """
        norm_local_emb = tf.nn.l2_normalize(local_vector, axis=1, name="norm_local_emb")
        norm_global_emb = tf.nn.l2_normalize(global_vector, axis=1, name="norm_global_emb")
        dim = global_vector.shape.as_list()[1]
        U_l = tf.get_variable(shape=[dim, dim], name='U_l')
        U_g = tf.get_variable(shape=[dim, dim], name='U_g')
        sig_t = tf.nn.sigmoid(tf.matmul(norm_local_emb, U_l) + tf.matmul(norm_global_emb, U_g))
        fused_vector = tf.multiply(sig_t, norm_local_emb) + tf.multiply(1-sig_t, norm_global_emb)
        
        return fused_vector
        
    def _build_hrne_att_encoder(self, seq_embeddings, seq_len, params):
        """
        Builds HRNE model with attention in 2nd layer
        """
        with tf.variable_scope('dynamic_seq2seq') as scope:
            #Define GRU cells for both the layers
            gru_1 = self._gru_cell(params, scope_name='gru_1')
         
            gru_1_len = params.stride
            num_timesteps = seq_embeddings.shape.as_list()[1]
            # Initialize the layer 1 and layer 2 hidden state.
            h_prev = tf.zeros([params.batch_size, params.num_units], name='h_init')
            
            output_gru_1=[]
            #Run the first layer of GRU and stack individual timesteps
            for step in range(num_timesteps):
                out, h_prev = gru_1(inputs=seq_embeddings[:,step,:], state=h_prev)
                output_gru_1.append(out)
			#Stack all the states and split the batch into individual samples
            stacked_states = tf.stack(output_gru_1, axis=1)
            state_dim=stacked_states.shape.as_list()[-1]
            batch_padded_state_vectors = tf.split(stacked_states, num_or_size_splits=params.batch_size, axis=0)
            batch_state_vectors=[]
            batch_strided_states=[]
            partitions=[]
            for index in range(len(batch_padded_state_vectors)):
                # Get the vectors corresponding to the actual length of the sample caption
                sliced_state = tf.squeeze(tf.slice(batch_padded_state_vectors[index], begin=[0,0,0], size=[1, seq_len[index], state_dim]))
                final_timestep = sliced_state[-1, :]
                batch_state_vectors.append(final_timestep)
                # Get the strided outputs. Strided slice includes the first timestep as well. Ignore that !!
                strided_sliced_state = tf.strided_slice(sliced_state[:-1, :], [0,0], [seq_len[index], state_dim], [params.stride, 1])[1:, :]
                # Infer the partitions given by strided slice op
                inferred_partitions = tf.cast(tf.ceil(tf.divide(seq_len[index]-1, params.stride)), tf.int32) -1
                partitions.append(inferred_partitions)
                batch_strided_states.append(strided_sliced_state)
                
            # Batch all the individual final timestep vectors back
            layer1_state_vectors = tf.stack(batch_state_vectors, axis=0)
            # Above looping caused dynamic shapes. 
            # Set the static shape to ensure rest of the graph builds with static shapes.
            layer1_state_vectors.set_shape([params.batch_size, state_dim])
            
            # Define shared attention matrix
            w_att = tf.get_variable(shape=[state_dim, state_dim], name='w_att', trainable=True)
            # Apply attention to strided states 
            batch_context_states=[]
            for strided_state in batch_strided_states:
                att_strided_state = tf.matmul(strided_state, w_att)
                max_pooled_att_state = tf.reduce_max(att_strided_state, axis=0)
                batch_context_states.append(max_pooled_att_state)
  
            # Stack the local context
            batch_context_vector = tf.stack(batch_context_states, axis=0, name='batch_context_vector')
            
            fused_vector = self._gated_fusion_unit(batch_context_vector, layer1_state_vectors)
        return fused_vector
                
class SCAN(object):
    """
    Implementation of SCAN
    """
    def __init__(self):
        pass
    
    def _gru_cell(self, params, scope_name):
        """
        Defines a GRU cell
        """
        gru_cell = tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.GRUCell(params.num_units, 
                                                                             name=scope_name, 
                                                                             reuse=tf.AUTO_REUSE), 
                                                                             input_keep_prob=(1.0 - params.dropout))
                                                                             
        return gru_cell
    
    def _build_text_encoder(self, inputs, params, seq_len):
        with tf.variable_scope('dynamic_seq2seq') as scope:
            #Define GRU cells for both the layers
            forward_cell = self._gru_cell(params, scope_name='fwd_gru')
            backward_cell = self._gru_cell(params, scope_name='bwd_gru')
            
            # Define the bi-directional rnn
            (output_fwd, output_bwd), (fwd_state, bwd_state) = tf.nn.bidirectional_dynamic_rnn(
                                                                            forward_cell,
                                                                            backward_cell,
                                                                            inputs,
                                                                            sequence_length=seq_len,
                                                                            dtype=tf.float32)
                              
            return (output_fwd+output_bwd)/2.0, (fwd_state+bwd_state)/2.0