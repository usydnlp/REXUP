import tensorflow as tf
import ops
from config import config

# https://www.tensorflow.org/beta/tutorials/text/transformer
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def gated_scaled_dot_product_attention(q, k, v, gate_dim, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    fc_gk = tf.keras.layers.Dense(gate_dim)(k)
    fc_gq = tf.keras.layers.Dense(gate_dim)(q)
    interaction = fc_gk * fc_gq
    M = tf.keras.layers.Dense(2, activation='sigmoid')(interaction)

    Mq = tf.tile(tf.expand_dims(M[:, :, :, 0], axis=-1), [1, 1, 1, q.shape[-1]])
    Mk = tf.tile(tf.expand_dims(M[:, :, :, 1], axis=-1), [1, 1, 1, q.shape[-1]])

    q = Mq * q
    k = Mk * k

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='selu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        if str(x.shape[1]) == "?":
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        else:
            x = tf.reshape(x, (batch_size, x.shape[1], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, is_self=False):
        batch_size = tf.shape(q)[0]
        # batch_size = q.shape[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        if is_self:
            scaled_attention, attention_weights = gated_scaled_dot_product_attention(q, k, v, 512, mask)
        else:
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        if str(scaled_attention.shape[1]) == "?":
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        else:
            concat_attention = tf.reshape(scaled_attention, (batch_size, scaled_attention.shape[1], self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)  # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list, split_size


def get_chunks(x, sizes):
    out = []
    begin = 0
    for s in sizes:
        if len(x.shape) > 2:
            y = x[:, :, begin:begin + s]  # tf.slice(x, [0, x.shape[1]], [x.shape[0], s])  # x.narrow(1, begin, s)
        else:
            y = x[:, begin:begin + s]
        out.append(y)
        begin += s
    return out


# Translate from pytorch version: https://github.com/Cadene/block.bootstrap.pytorch
class Block(tf.keras.layers.Layer):

    def __init__(self,
                 # input_dims,
                 output_dim,
                 mm_dim=1600,
                 chunks=20,
                 rank=15,
                 shared=False,
                 dropout_input=0.,
                 dropout_pre_lin=0.,
                 dropout_output=0.):
        super(Block, self).__init__()
        # self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = tf.keras.layers.Dense(mm_dim)  # nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = tf.keras.layers.Dense(mm_dim)  # nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list, self.split_size = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = tf.keras.layers.Dense(size * rank)  # nn.Linear(size, size * rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = tf.keras.layers.Dense(size * rank)  # nn.Linear(size, size * rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = merge_linears0  # nn.ModuleList(merge_linears0)
        self.merge_linears1 = merge_linears1  # nn.ModuleList(merge_linears1)
        self.linear_out = tf.keras.layers.Dense(output_dim)  # nn.Linear(mm_dim, output_dim)
        # self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def call(self, x, training, seqlen=0):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = tf.shape(x1)[0]  # x1.shape[0]
        if self.dropout_input > 0:
            x0 = tf.keras.layers.Dropout(self.dropout_input)(x0, training=training)  # F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = tf.keras.layers.Dropout(self.dropout_input)(x1, training=training)  # F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)),
                                    self.merge_linears0,
                                    self.merge_linears1):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]

            m = m0(x0_c) * m1(x1_c)  # bsize x split_size*rank
            if seqlen != 0:
                m = tf.reshape(m, (bsize, seqlen, self.rank, self.split_size))
            else:
                m = tf.reshape(m, (bsize, self.rank, self.split_size))
            z = tf.reduce_sum(m, axis=-1)  # torch.sum(m, 1)
            z = tf.math.sqrt(tf.nn.relu(z)) - tf.math.sqrt(tf.nn.relu(-z))  # torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            # z = tf.keras.utils.normalize(z, order=2)  # F.normalize(z, p=2)
            z = tf.math.l2_normalize(z, axis=-1)
            # print("zs", z.shape, "m", m.shape)
            zs.append(z)

        z = tf.concat(zs, axis=-1)  # torch.cat(zs, 1)
        print("z", z.shape)
        if self.dropout_pre_lin > 0:
            z = tf.keras.layers.Dropout(self.dropout_input)(z, training=training)  # F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = tf.keras.layers.Dropout(self.dropout_input)(z, training=training)  # F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class MemEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(MemEncoder, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha3 = MultiHeadAttention(d_model, num_heads)
        # self.mha4 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm5 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)
        # self.dropout5 = tf.keras.layers.Dropout(rate)

        self.block1 = Block(512,
                            mm_dim=1600,
                            chunks=20,
                            rank=15,
                            shared=False,
                            dropout_input=0.1)

        self.block2 = Block(512,
                            mm_dim=1600,
                            chunks=20,
                            rank=15,
                            shared=False,
                            dropout_input=0.1)

    def call(self, mem_state, control_state, attend_words, knowledge_base, training,
             look_ahead_mask, padding_mask):
        kb_maxLength = tf.shape(knowledge_base)[-2]
        mask = (tf.to_float(tf.logical_not(tf.sequence_mask(padding_mask, kb_maxLength))))
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.expand_dims(mask, axis=1)

        attn1, attn_weights_block1 = self.mha1(mem_state, mem_state, mem_state, look_ahead_mask, is_self=True)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + mem_state)

        attn2, attn_weights_block2 = self.mha2(knowledge_base, knowledge_base, out1, None, is_self=False)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        control_state = tf.reshape(control_state, (-1, control_state.shape[2]))
        out2 = tf.reshape(out2, (-1, out2.shape[2]))
        interactions = self.block2([out2, control_state], training=training)
        interactions = tf.expand_dims(interactions, 1)

        # print("##########", control_state.shape, out2.shape, interactions.shape)
        interactions = ops.activations[config.readCtrlAct](interactions)

        attn3, attn_weights_block3 = self.mha3(knowledge_base, knowledge_base, interactions, None)  # (batch_size, target_seq_len, d_model)
        attn3 = self.dropout3(attn3, training=training)
        out3 = self.layernorm3(attn3 + interactions)  # (batch_size, target_seq_len, d_model)

        # out3 = interactions

        ffn_output = self.ffn(out3)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm4(ffn_output + out3)  # (batch_size, target_seq_len, d_model)

        return out3  # , attn_weights_block1, attn_weights_block2, attn_weights_block3
