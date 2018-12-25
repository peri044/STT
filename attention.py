import tensorflow as tf
import numpy as np
import pdb

def cosine_sim(query, ref, axis=2):
    
    numerator = tf.reduce_sum(tf.multiply(query, ref), axis=axis)
    query_norm = tf.norm(query, axis=axis)
    ref_norm = tf.norm(ref, axis=axis)
    
    return numerator/tf.maximum(query_norm*ref_norm, 1e-8)

def compute_attention(query, context, params):
    """
    query: (B x n_query x d)
    context: (B x n_context x d)
    """
    batch_size_q, num_words_q = query.shape.as_list()[0], query.shape.as_list()[1]
    batch_size_c, num_regions_c = context.shape.as_list()[0], context.shape.as_list()[1]

    attn = tf.matmul(context, query, transpose_b=True) # B x n_context x n_query
    
    # clipped leaky l2 norm
    clip_attn = tf.nn.leaky_relu(attn, alpha=0.1) # B x n_context x n_query
    norm_attn = tf.nn.l2_normalize(clip_attn, axis=2) # B x n_context x n_query
    attn_transpose = tf.transpose(norm_attn, [0, 2, 1]) # B x n_query x n_context
    soft_attn = tf.nn.softmax(attn_transpose*params.lambda_1) 
    soft_attn_transpose = tf.transpose(soft_attn, [0, 2, 1]) # B x n_context x n_query
    
    context_transpose = tf.transpose(context, [0, 2, 1]) # B x d x n_context
    weighted_attn = tf.matmul(context_transpose, soft_attn_transpose) # B x d x n_query
    weighted_attn_context = tf.transpose(weighted_attn, [0, 2, 1]) # B x n_query x d
    
    return weighted_attn_context, soft_attn_transpose
    
def t2i_attention(image_embeddings, text_embeddings, seq_len, params):
    """
    Text-to-Image Attention
    """
    n_image = image_embeddings.shape.as_list()[0]
    n_caption = text_embeddings.shape.as_list()[0]

    similarities=[]
    for i in range(n_caption):
        n_word = seq_len[i]
        cap_i = tf.expand_dims(text_embeddings[i, :n_word, :], 0)
        tiled_cap_i = tf.tile(cap_i, [n_image, 1, 1])
        
        weighted_attn_context, sim_matrix = compute_attention(tiled_cap_i, image_embeddings, params)
        # row_sim --> B x n_word
        row_sim = cosine_sim(tiled_cap_i, weighted_attn_context) # B x n_word x d , B x n_word x d
        row_sim = tf.reduce_mean(row_sim, axis=1)
        similarities.append(row_sim)

    sim_matrix = tf.stack(similarities, axis=1)

    return sim_matrix
    
def i2t_attention(image_embeddings, text_embeddings, seq_len, params):
    """
    Image-to-Text Attention
    """
    n_image = image_embeddings.shape.as_list()[0]
    n_caption = text_embeddings.shape.as_list()[0]

    similarities=[]
    for i in range(n_caption):
        n_word = seq_len[i]
        cap_i = tf.expand_dims(text_embeddings[i, :n_word, :], 0)
        tiled_cap_i = tf.tile(cap_i, [n_image, 1, 1])
        
        weighted_attn_context, _ = compute_attention(image_embeddings, tiled_cap_i, params) # Weighted sentence vector
        # row_sim --> B x n_word
        row_sim = cosine_sim(image_embeddings, weighted_attn_context, axis=2) # B x n_word x d , B x n_word x d
        row_sim = tf.reduce_mean(row_sim, axis=1)
        similarities.append(row_sim)

    sim_matrix = tf.stack(similarities, axis=1)

    return sim_matrix     

# def compute_sample_attention():

def compute_para_attention(query, context, params):
    """
    query: (B x n_query x d)
    context: (B x n_context x d)
    """
    batch_size_q, num_words_q = query.shape.as_list()[0], query.shape.as_list()[1]
    batch_size_c, num_regions_c = context.shape.as_list()[0], context.shape.as_list()[1]

    attn = tf.matmul(context, query, transpose_b=True) # B x n_context x n_query
    
    # clipped leaky l2 norm
    clip_attn = tf.nn.leaky_relu(attn, alpha=0.1) # B x n_context x n_query
    norm_attn = tf.nn.l2_normalize(clip_attn, axis=2) # B x n_context x n_query
    attn_transpose = tf.transpose(norm_attn, [0, 2, 1]) # B x n_query x n_context
    soft_attn = tf.nn.softmax(attn_transpose*params.lambda_1) 
    soft_attn_transpose = tf.transpose(soft_attn, [0, 2, 1]) # B x n_context x n_query
    
    context_transpose = tf.transpose(context, [0, 2, 1]) # B x d x n_context
    weighted_attn = tf.matmul(context_transpose, soft_attn_transpose) # B x d x n_query
    weighted_attn_context = tf.transpose(weighted_attn, [0, 2, 1]) # B x n_query x d
    
    return weighted_attn_context, soft_attn_transpose    

def para_attention(enc_embeddings, dec_embeddings, enc_len, dec_len, params):
    """
    Text-Text attention
    """
    batch_size = enc_embeddings.shape.as_list()[0]
    similarities=[]
    dec_ind_embeddings=[]
    for j in range(batch_size):
        n_word = dec_len[j]
        dec_ind_embeddings.append(tf.expand_dims(dec_embeddings[j, :n_word, :], 0))
    whole_sim=[]
    for i in range(batch_size):
        n_enc_word = enc_len[i]
        n_dec_word = dec_len[i]
        
        # Replicate the enc_embeddings which is the query
        enc_cap_i = tf.expand_dims(enc_embeddings[i, :n_enc_word, :], 0)
        
        individual_row_sim=[]
        for sample in dec_ind_embeddings:
            weighted_attn_context, _ = compute_para_attention(enc_cap_i, sample, params)
            row_sim = cosine_sim(enc_cap_i, weighted_attn_context, axis=2) # B x n_word x d , B x n_word x d
            mean_row_sim = tf.reduce_mean(row_sim)
            individual_row_sim.append(mean_row_sim)

        whole_sim.append(individual_row_sim)
    
    sim_matrix = tf.stack(whole_sim, axis=1)
    
    return sim_matrix
        
        
        
    