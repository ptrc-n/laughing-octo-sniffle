import tensorflow as tf
from tensorflow.keras import layers
import silence_tensorflow.auto  # silence TF warnings


# create a padding mask for sequences of different lengths
def create_padding_mask(seq, pad_symbol=0):
    """
    Args:
    -----
        seq: a tensor of shape (batch_size, seq_len) defaults to 0
    
    Returns:
    --------
        mask: a tensor of shape (batch_size, 1, 1, seq_len)
    """
    seq = tf.cast(tf.math.equal(seq, pad_symbol), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# create look ahead mask
def create_look_ahead_mask(size):
    """
    Args:
    -----
        size: specifies the sequence length

    Returns:
    --------
        mask: of dimensions (seq_len, seq_len)
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


# define scaled_dot_product_attention
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
    -----
        q: queries of shape (batch_size, seq_len_q, depth_qk)
        k: keys of shape (batch_size, seq_len_k, depth_qk)
        v: values of shape (batch_size, seq_len_v, depth_v)
        mask: mask of shape (batch_size, 1, 1, seq_len_q) defaults to None
    
    Returns:
    --------
        output: of shape (batch_size, seq_len_q, d_model)
        attention_weights: of shape (batch_size, seq_len_q, d_model)
    """
    # get the product of the query and key vectors
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale the product
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # apply the attention weights to the value vectors
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


#create multi headed attention layer
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, dropout):
        """
        Args:
        -----
            d_model: dimension of the model
            num_heads: number of heads
            dropout: dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        split the last dimension into (num_heads, depth)
        Args:
        -----
            x: of shape (batch_size, seq_len, d_model)
            batch_size: of shape (batch_size)
        
        Returns:
        --------
            x: of shape (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(
            q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(
            k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(
            v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(
            concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


# create point wise feed forward network
def point_wise_feed_forward_network(d_model, dff):
    """
    Args:
    -----
        d_model: dimension of the model
        dff: dimension of the feed forward network
    
    Returns:
    --------
        output: of shape (batch_size, seq_len, d_model)
    """
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# create a network to reduce the dimension of the input
def dim_reduction_network(dim_input_data, dim_output_data):
    """
    Args:
    -----
        dim_input_data: dimension of the input
        dim_output_data: dimension of the output must be less than dim_input_data
    
    Returns:
    --------
        output: of shape (dim_output_data)
    """
    middle_layer_dim = dim_output_data + (dim_input_data - dim_output_data) / 2
    return tf.keras.Sequential([
        layers.Dense(dim_output_data, activation='linear'),
        layers.Dense(middle_layer_dim, activation='relu'),
        layers.Dense(dim_input_data, activation='linear'),
    ])


# define prototype of encorder layer
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Args:
        -----
            d_model: dimension of the model
            num_heads: number of attention heads
            dff: dimension of the feed forward network
            rate: dropout rate
        """
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, rate)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x +
                               attn_output)  # (batch_size, seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 +
                               ffn_output)  # (batch_size, seq_len, d_model)
        return out2


# create decoder layer
class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Args:
        -----
            d_model: dimension of the model
            num_heads: number of attention heads
            dff: dimension of the feed forward network
            rate: dropout rate
        """
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, rate)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(
        self,
        x,
        enc_output,
        training,
        look_ahead_mask,
        padding_mask,
    ):
        """ 
        Args:
        -----
            x: of shape (batch_size, seq_len, d_model)
            enc_output: of shape (batch_size, seq_len, d_model)
            training: boolean
            look_ahead_mask: of shape (batch_size, seq_len, seq_len)
            padding_mask: of shape (batch_size, 1, seq_len)
        
        Returns:
        --------
            output: of shape (batch_size, seq_len, d_model)
            attention_weights_block1: of shape (batch_size, num_heads, seq_len_q, seq_len_k)
            attention_weights_block2: of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                               padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output +
                               out2)  # (batch_size, seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# create a prototype for an encoder
class Encoder(layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_dimensions,
        rate=0.1,
    ):
        """
        Args:
        -----
            num_layers: number of layers
            d_model: dimension of the model
            num_heads: number of attention heads
            dff: dimension of the feed forward network
            input_dimensions: dimension of the input vector (max_active_regions, seq_len)
            rate: dropout rate
        """
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.embeddings = [
            dim_reduction_network(input_dimensions[1], d_model)
            for _ in range(input_dimensions[0])
        ]
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        Args:
        -----
            x: of shape (max_active_regions, seq_len)
            training: boolean
            mask: of shape (max_active_regions, seq_len, seq_len)
        
        Returns:
        --------
            output: of shape (max_active_regions, seq_len, d_model)
            attention_weights: of shape (max_active_regions, num_heads, seq_len_q, seq_len_k)
        """
        # adding embedding
        embeddings = [[
            self.embeddings[ar](x[timestep][ar])
            for ar in range(len(self.embeddings))
        ] for timestep in range(len(x))]
        # (timesteps, max_active_regions, d_model)

        # reduce dimensionality
        x = [layers.Add(embeddings[timestep]) for timestep in range(len(x))]
        # (n_timesteps, d_model)

        x = [
            self.dropout(x[timestep], training=training)
            for timestep in range(len(x))
        ]
        # (n_timesteps, d_model)

        for i in range(self.num_layers):
            x, _ = self.enc_layers[i](x, training, mask)

        return x


# create a prototype for a decoder
class Decoder(layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_dimensions,
        rate=0.1,
    ):
        """
        Args:
        -----
            num_layers: number of layers
            d_model: dimension of the model
            num_heads: number of attention heads
            dff: dimensions of point to point ff network
            target_dimensions: dimension of the target vector
            rate: dropout rate
        """
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = layers.Embedding(target_dimensions, d_model)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = layers.Dropout(rate)

    def call(
        self,
        x,
        enc_output,
        training,
        look_ahead_mask,
        padding_mask,
    ):
        """
        Args:
        -----
            x: of shape (batch_size, target_seq_len, d_model)
            enc_output: of shape (batch_size, seq_len, d_model)
            training: boolean
            look_ahead_mask: of shape (batch_size, target_seq_len, target_seq_len)
            padding_mask: of shape (batch_size, 1, target_seq_len)
        
        Returns:
        --------
            output: of shape (batch_size, target_seq_len, d_model)
            attention_weights: of shape (batch_size, num_heads, target_seq_len_q, target_seq_len_k)
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attention_weights[f"layer{i}"] = self.dec_layers[i](
                x,
                enc_output,
                training,
                look_ahead_mask,
                padding_mask,
            )

        return x, attention_weights


# create a prototype for a transformer
class Transformer(layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_dimensions,
        target_dimensions,
        rate=0.1,
    ):
        """
        Args:
        -----
            num_layers: number of layers
            d_model: dimension of the model
            num_heads: number of attention heads
            dff: dimensions of point to point ff network
            input_dimensions: dimension of the input vector (n_timesteps, max_active_regions, seq_len)
            target_dimensions: dimension of the target vector (n_timesteps, output_dim)
            rate: dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_dimensions, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_dimensions, rate)

        self.final_layer = layers.Dense(target_dimensions)

    def create_masks(self, inp, tar):
        """
        Args:
        -----
            inp: of shape (batch_size, seq_len)
            tar: of shape (batch_size, target_seq_len)
        
        Returns:
        --------
            enc_padding_mask: of shape (batch_size, 1, seq_len)
            look_ahead_mask: of shape (batch_size, target_seq_len, target_seq_len)
            dec_padding_mask: of shape (batch_size, 1, target_seq_len)
        """
        # padding mask
        enc_padding_mask = create_padding_mask(inp)

        # look ahead mask
        dec_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    def call(self, inputs, training):
        """
        Args:
        -----
            inputs: of shape [(batch_size, seq_len), (batch_size, target_seq_len)]
            training: boolean
        
        Returns:
        --------
            output: of shape (batch_size, target_seq_len, target_dimensions)
            attention_weights: of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            inp,
            tar,
        )

        enc_output = self.encoder(
            inp,
            training,
            enc_padding_mask,
        )

        dec_output, attention_weights = self.decoder(
            tar,
            enc_output,
            training,
            look_ahead_mask,
            dec_padding_mask,
        )

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
