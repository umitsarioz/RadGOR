import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model,load_model
from pickle import load
import os


class PositionEncoder:
    def aci_hesapla(self, pos, i, embed_dim):
        aci_orani = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * aci_orani

    def toDecoder(self, pos, embed_dim):
        aci_orani = self.aci_hesapla(np.arange(pos)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)

        aci_orani[:, 0::2] = np.sin(aci_orani[:, 0::2])
        aci_orani[:, 1::2] = np.cos(aci_orani[:, 1::2])
        pos_encoding = aci_orani[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def toEncoder(self, satir, sutun, embed_dim):
        assert embed_dim % 2 == 0
        satir_pos = np.repeat(np.arange(satir), sutun)[:, np.newaxis]
        sutun_pos = np.repeat(np.expand_dims(np.arange(sutun), 0), satir, axis=0).reshape(-1, 1)

        aci_orani_satir = self.aci_hesapla(satir_pos, np.arange(embed_dim // 2)[np.newaxis, :], embed_dim // 2)
        aci_orani_sutun = self.aci_hesapla(sutun_pos, np.arange(embed_dim // 2)[np.newaxis, :], embed_dim // 2)

        aci_orani_satir[:, 0::2] = np.sin(aci_orani_satir[:, 0::2])
        aci_orani_satir[:, 1::2] = np.cos(aci_orani_satir[:, 1::2])
        aci_orani_sutun[:, 0::2] = np.sin(aci_orani_sutun[:, 0::2])
        aci_orani_sutun[:, 1::2] = np.cos(aci_orani_sutun[:, 1::2])
        pos_encoding = np.concatenate([aci_orani_satir, aci_orani_sutun], axis=1)[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, nheads):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.embed_dim = embed_dim
        assert embed_dim % self.nheads == 0
        self.depth = embed_dim // self.nheads
        self.wq = tf.keras.layers.Dense(embed_dim)
        self.wk = tf.keras.layers.Dense(embed_dim)
        self.wv = tf.keras.layers.Dense(embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.nheads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, embed_dim)
        k = self.wk(k)  # (batch_size, seq_len, embed_dim)
        v = self.wv(v)  # (batch_size, seq_len, embed_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, nheads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, nheads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, nheads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, nheads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.embed_dim))  # (batch_size, seq_len_q, embed_dim)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, embed_dim)
        return output, attention_weights


def point_wise_feed_forward_network(embed_dim, ffn_dim):
    return tf.keras.Sequential([tf.keras.layers.Dense(ffn_dim, activation='relu'),
                                tf.keras.layers.Dense(embed_dim)])  # (batch_size, seq_len, embed_dim)])


class EncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_dim, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(embed_dim, ffn_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, embed_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, embed_dim)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embed_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embed_dim)
        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, embed_dim, nheads, ffn_dim, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(embed_dim, nheads)
        self.mha2 = MultiHeadAttention(embed_dim, nheads)

        self.ffn = point_wise_feed_forward_network(embed_dim, ffn_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, embed_dim)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, embed_dim)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, embed_dim)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, embed_dim)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, nheads, ffn_dim, row_size, col_size, rate=0.1):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(self.embed_dim, activation='relu')
        self.pos_encoding = PositionEncoder().toEncoder(row_size, col_size, self.embed_dim)

        self.enc_layers = [EncoderLayer(embed_dim, nheads, ffn_dim, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len(H*W), embed_dim)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, embed_dim)


class Decoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, nheads, ffn_dim, vocabsize, rate=0.1):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocabsize, embed_dim)
        self.pos_encoding = PositionEncoder().toDecoder(vocabsize, embed_dim)

        self.dec_layers = [DecoderLayer(embed_dim, nheads, ffn_dim, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, embed_dim)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class Transformer(Model):
    def __init__(self, num_layers, embed_dim, nheads, ffn_dim, row_size, col_size, vocabsize, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, embed_dim, nheads, ffn_dim, row_size, col_size, rate)
        self.decoder = Decoder(num_layers, embed_dim, nheads, ffn_dim, vocabsize, rate)
        self.final_layer = tf.keras.layers.Dense(vocabsize)

    def call(self, inp, tar, training, look_ahead_mask=None, dec_padding_mask=None, enc_padding_mask=None):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, embed_dim  )
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, vocabsize)
        return final_output


CURR_DIR = os.getcwd()
tokenizer_path = os.path.join(CURR_DIR,'modeller','mytknzr.pkl')
tokenizerMap = load(open(tokenizer_path,'rb'))
tokenizer = tokenizerMap['tknzr']
VOCABSIZE = len(tokenizer.index_word)+1
NUM_LAYER = 6 # encoder-decoder blok tekrar sayısı
EMBED_DIM = 512 # embedding dimension
FFN_DIM = 3072 #feed-forward network dimension
NHEADS = 8 # kafa sayısı
ROW_SIZE = 8 # encoder-input- hizalama satir sayisi
COL_SIZE = 8 # encoder-input- hizalama sutun sayisi
DROPOUT_RATE = 0.1 # dropout oranı

def loadPredictionModel():
    baseTransformerModel = Transformer(NUM_LAYER, EMBED_DIM, NHEADS, FFN_DIM,ROW_SIZE,COL_SIZE, VOCABSIZE, rate=DROPOUT_RATE)
    return baseTransformerModel

def loadExtractModel(extract_model_path):
    return load_model(extract_model_path)

def loadTokenizer(filepath):
    tknzr,maxlen = list(load(open(filepath,'rb')).values())
    return tknzr,maxlen