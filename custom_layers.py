import tensorflow as tf
from keras import backend as k
from keras.engine.topology import Layer
from keras.regularizers import l2


# input shape: N, sen_num, num_steps, gru_size
# output shape: N, sen_num, gru_size
class WordLevelAttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WordLevelAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # print(input_shape)
        self.attention_w = self.add_weight(name='attention_omega', shape=(input_shape[-1], 1),
                                           initializer='glorot_uniform', trainable=True, regularizer=l2(0.0001))
        super(WordLevelAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # print(k.int_shape(inputs))
        sen_num = k.int_shape(inputs)[1]
        inputs = k.reshape(inputs, shape=(-1, k.int_shape(inputs)[-2], k.int_shape(inputs)[-1]))
        self.total_num = k.shape(inputs)[0]
        self.num_steps = k.shape(inputs)[1]
        self.gru_size = k.shape(inputs)[2]
        m = k.reshape(k.tanh(inputs), shape=[self.total_num * self.num_steps, self.gru_size])  # shape: total_num*num_steps, gru_size
        tmp = k.reshape(k.dot(m, self.attention_w), shape=[self.total_num, self.num_steps])
        alpha = k.reshape(k.softmax(tmp), shape=[self.total_num, 1, self.num_steps])
        r = k.batch_dot(alpha, inputs)
        attention_r = k.reshape(r, shape=(-1, sen_num, self.gru_size))
        # print(type(attention_r))
        return attention_r

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            assert inputs.shape == mask.shape
            mask = mask[:, :, -1]
        return mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[-1])


# input shape: sample, sen_num, gru_size
# output shape: sample, num_classes
class SentenceLevelAttentionLayer(Layer):
    def __init__(self, num_classes, **kwargs):
        self.supports_masking = True
        self.num_classes = num_classes
        super(SentenceLevelAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gru_size = input_shape[-1]
        self.sen_a = self.add_weight(name='attention_A', shape=(self.gru_size,), initializer='glorot_uniform',
                                     trainable=True, regularizer=l2(0.0001))
        self.sen_r = self.add_weight(name='query_r', shape=(self.gru_size, 1), initializer='glorot_uniform',
                                     trainable=True, regularizer=l2(0.0001))
        self.relation_embedding = self.add_weight(name='relation_embedding', shape=(self.num_classes, self.gru_size),
                                                  initializer='glorot_uniform', trainable=True, regularizer=l2(0.0001))
        self.sen_d = self.add_weight(name='bias_d', shape=(self.num_classes,), initializer='glorot_uniform',
                                     trainable=True, regularizer=l2(0.0001))

        super(SentenceLevelAttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs * mask
            sen_repre = k.tanh(inputs)  # sample, sen_num, gru_size
            # print(inputs.shape)
            # print(self.sen_a.shape)
            e = k.dot(sen_repre * self.sen_a, self.sen_r)  # sample, sen_num, 1
            e = k.squeeze(e, axis=-1)  # sample, sen_num
            mask = (e != 0)
            sen_alpha = self.masked_softmax(e, mask)  # sample, sen_num
            # sen_alpha = k.exp(e)
            # sen_alpha /= k.cast(k.sum(sen_alpha, axis=1, keepdims=True) + k.epsilon(), k.floatx())  # sample, sen_num, 1
            # print(sen_alpha.shape)
            sen_s = k.batch_dot(sen_alpha, sen_repre)  # sample, gru_size
            # print(sen_s.shape)
            sen_out = k.dot(sen_s, k.transpose(self.relation_embedding)) + self.sen_d  # sample, num_classes
            sen_out = k.softmax(sen_out)
            sen_out = k.reshape(sen_out, shape=(-1, self.num_classes))
            # print(sen_out.shape)
        else:
            sen_repre = k.tanh(inputs)  # sample, sen_num, gru_size
            # print(inputs.shape)
            # print(self.sen_a.shape)
            e = k.dot(sen_repre * self.sen_a, self.sen_r)  # sample, sen_num, 1
            e = k.squeeze(e, axis=-1) # sample, sen_num
            sen_alpha = k.exp(e)
            sen_alpha /= k.cast(k.sum(sen_alpha, axis=1, keepdims=True) + k.epsilon(), k.floatx())  # sample, sen_num
            # print(sen_alpha.shape)
            sen_s = k.batch_dot(sen_alpha, sen_repre)  # sample, gru_size
            # print(sen_s.shape)
            sen_out = k.dot(sen_s, k.transpose(self.relation_embedding)) + self.sen_d  # sample, num_classes
            sen_out = k.softmax(sen_out)
            sen_out = k.reshape(sen_out, shape=(-1, self.num_classes))
            # print(sen_out.shape)
        return sen_out

    def masked_softmax(self, vec, mask, dim=1):
        masked_vec = vec * mask
        max_vec = k.max(masked_vec, axis=dim, keepdims=True)[0]
        exps = k.exp(masked_vec - max_vec)
        masked_exps = exps * mask
        masked_sums = k.sum(masked_exps, axis=dim, keepdims=True)
        zeros = (masked_sums == 0)
        masked_sums += zeros
        return masked_exps / masked_sums

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

