import tensorflow as tf
from keras import backend as k
from keras.engine.topology import Layer
import numpy as np


# input shape: total_num, num_steps, gru_size
# output shape: total_num, gru_size
class WordLevelAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(WordLevelAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # print(input_shape)
        self.attention_w = self.add_weight(name='attention_omega', shape=(input_shape[-1], 1),
                                           initializer='glorot_uniform', trainable=True)
        super(WordLevelAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # print(k.int_shape(inputs))
        self.total_num =k.shape(inputs)[0]
        self.num_steps = k.shape(inputs)[1]
        self.gru_size = k.shape(inputs)[2]
        m = k.reshape(k.tanh(inputs), shape=[self.total_num * self.num_steps, self.gru_size])  # shape: total_num*num_steps, gru_size
        tmp = k.reshape(k.dot(m, self.attention_w), shape=[self.total_num, self.num_steps])
        alpha = k.reshape(k.softmax(tmp), shape=[self.total_num, 1, self.num_steps])
        r = k.batch_dot(alpha, inputs)
        attention_r = k.reshape(r, shape=[self.total_num, self.gru_size])
        # print(type(attention_r))
        return attention_r

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# input shape: sample, sen_num, gru_size
# output shape: sample, gru_size
class SentenceLevelAttentionLayer(Layer):
    def __init__(self, big_num, num_classes, **kwargs):
        self.big_num = big_num
        self.num_classes = num_classes
        super(SentenceLevelAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gru_size = input_shape[-1]
        self.sen_a = self.add_weight(name='attention_A', shape=(self.gru_size,), initializer='glorot_uniform',
                                     trainable=True)
        self.sen_r = self.add_weight(name='query_r', shape=(self.gru_size, 1), initializer='glorot_uniform',
                                     trainable=True)
        self.relation_embedding = self.add_weight(name='relation_embedding', shape=(self.num_classes, self.gru_size),
                                                  initializer='glorot_uniform', trainable=True)
        self.sen_d = self.add_weight(name='bias_d', shape=(self.num_classes,), initializer='glorot_uniform', trainable=True)

        super(SentenceLevelAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sen_repre = k.tanh(inputs)  # sample, sen_num, gru_size
        # print(sen_repre.shape)
        e = k.dot(tf.multiply(sen_repre, self.sen_a), self.sen_r)  # sample, sen_num
        sen_alpha = k.exp(e)
        sen_alpha /= k.cast(k.sum(sen_alpha, axis=1, keepdims=True) + k.epsilon(), k.floatx())  # sample, sen_num
        sen_alpha = k.reshape(sen_alpha, k.shape(sen_alpha)[0:2])
        # print(sen_alpha.shape)
        sen_s = k.batch_dot(sen_alpha, sen_repre)  # sample, gru_size
        # print(sen_s.shape)
        sen_out = tf.add(k.dot(sen_s, k.transpose(self.relation_embedding)), self.sen_d)
        sen_out = k.softmax(sen_out)
        # print(sen_out.shape)
        return sen_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

