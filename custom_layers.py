from keras import backend as k
from keras.engine.topology import Layer
from keras.regularizers import l2


# input shape: N, sen_num, steps, d
# output shape: N * sen_num, steps, d
# input mask shape: N, sen_num, steps
class FlattenLayer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(FlattenLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FlattenLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = k.reshape(inputs, shape=(-1, k.int_shape(inputs)[-2], k.int_shape(inputs)[-1]))
        return inputs

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask
        return k.reshape(mask, (-1, k.int_shape(inputs)[-2]))

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-2], input_shape[-1])


# input shape: N * sen_num, steps, gru
# output shape: N, sen_num, steps, gru
# input mask shape: N * sen_num, steps
class ReshapeLayer(Layer):
    def __init__(self, sen_num, **kwargs):
        self.supports_masking = True
        self.sen_num = sen_num
        super(ReshapeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReshapeLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = k.reshape(inputs, shape=(-1, self.sen_num, k.int_shape(inputs)[-2], k.int_shape(inputs)[-1]))
        return inputs

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask
        return k.reshape(mask, (-1, self.sen_num, k.int_shape(mask)[-1]))

    def compute_output_shape(self, input_shape):
        return (None, self.sen_num, input_shape[-2], input_shape[-1])


# input shape: N, sen_num, num_steps, gru_size
# output shape: N, sen_num, gru_size
# input mask shape: N, sen_num, shape
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
        # print(mask is not None)
        sen_num = k.int_shape(inputs)[1]
        inputs = k.reshape(inputs, shape=(-1, k.int_shape(inputs)[-2], k.int_shape(inputs)[-1]))
        self.total_num = k.shape(inputs)[0]
        self.num_steps = k.shape(inputs)[1]
        self.gru_size = k.shape(inputs)[2]
        m = k.reshape(k.tanh(inputs),
                      shape=[self.total_num * self.num_steps, self.gru_size])  # shape: total_num*num_steps, gru_size
        tmp = k.reshape(k.dot(m, self.attention_w), shape=[self.total_num, self.num_steps])
        alpha = k.reshape(k.softmax(tmp), shape=[self.total_num, 1, self.num_steps])
        r = k.batch_dot(alpha, inputs)
        attention_r = k.reshape(r, shape=(-1, sen_num, self.gru_size))
        # print(type(attention_r))
        return attention_r

    def compute_mask(self, inputs, mask=None):
        # print(mask.shape)
        return mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[-1])


# input shape: sample, sen_num, gru_size
# output shape: sample, num_classes
# input mask shape: sample, sen_num, num_steps
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
            mask = k.dot(k.cast(mask, k.floatx()), k.ones(shape=(k.int_shape(mask)[-1], k.int_shape(inputs)[-1])))
            inputs = inputs * k.cast(mask, k.floatx())
            sen_repre = k.tanh(inputs)  # sample, sen_num, gru_size
            e = k.dot(sen_repre * self.sen_a, self.sen_r)  # sample, sen_num, 1
            e = k.squeeze(e, axis=-1)  # sample, sen_num
            mask = (e != 0)
            sen_alpha = self.masked_softmax(e, mask)  # sample, sen_num)
            sen_s = k.batch_dot(sen_alpha, sen_repre)  # sample, gru_size
            sen_out = k.dot(sen_s, k.transpose(self.relation_embedding)) + self.sen_d  # sample, num_classes
            sen_out = k.softmax(sen_out)
            sen_out = k.reshape(sen_out, shape=(-1, self.num_classes))
        else:
            sen_repre = k.tanh(inputs)  # sample, sen_num, gru_size
            e = k.dot(sen_repre * self.sen_a, self.sen_r)  # sample, sen_num, 1
            e = k.squeeze(e, axis=-1)  # sample, sen_num
            sen_alpha = k.exp(e)
            sen_alpha /= k.cast(k.sum(sen_alpha, axis=1, keepdims=True) + k.epsilon(), k.floatx())  # sample, sen_num
            sen_s = k.batch_dot(sen_alpha, sen_repre)  # sample, gru_size
            sen_out = k.dot(sen_s, k.transpose(self.relation_embedding)) + self.sen_d  # sample, num_classes
            sen_out = k.softmax(sen_out)
            sen_out = k.reshape(sen_out, shape=(-1, self.num_classes))
        return sen_out

    def masked_softmax(self, vec, mask):
        vec = vec * mask
        e = k.exp(vec)
        e = e * mask
        return e / k.sum(e, axis=-1, keepdims=True)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)
