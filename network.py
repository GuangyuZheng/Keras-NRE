import tensorflow as tf
from keras import backend as k
from keras.layers import Embedding, concatenate, GRU, Bidirectional
from custom_layers import WordLevelAttentionLayer, SentenceLevelAttentionLayer


class Settings(object):
    def __init__(self):
        self.vocab_size = 114042
        self.num_steps = 70
        self.num_epochs = 3
        self.num_classes = 53
        self.gru_size = 230  # sentence embedding
        self.keep_prob = 0.5  # dropout probability
        self.num_layers = 1
        self.pos_size = 5  # position embeddings
        self.pos_num = 123
        # the number of entity pairs of each batch during training or testing
        self.big_num = 50


class BGRU_2ATT:
    def __init__(self, word_embeddings, settings):
        self.num_steps = settings.num_steps
        self.num_classes = settings.num_classes
        self.word_embeddings = word_embeddings
        self.pos_num = settings.pos_num
        self.pos_size = settings.pos_size
        self.gru_size = settings.gru_size
        self.keep_prob = settings.keep_prob
        self.big_num = settings.big_num

        # embedding
        words_embedding_layer = Embedding(len(self.word_embeddings), len(self.word_embeddings[0]), weights=[self.
                                          word_embeddings], trainable=False)
        pos1_embedding_layer = Embedding(self.pos_num, self.pos_size, embeddings_initializer='glorot_uniform',
                                         trainable=True)
        pos2_embedding_layer = Embedding(self.pos_num, self.pos_size, embeddings_initializer='glorot_uniform',
                                         trainable=True)

        BGRU_layer = Bidirectional(GRU(units=self.gru_size, return_sequences=True, dropout=1 - self.keep_prob),
                                   merge_mode='sum')

        self.input_words = k.placeholder(dtype='int32', shape=[None, self.num_steps], name='input_words')
        self.input_pos1 = k.placeholder(dtype='int32', shape=[None, self.num_steps], name='input_pos1')
        self.input_pos2 = k.placeholder(dtype='int32', shape=[None, self.num_steps], name='input_pos2')
        self.input_y = k.placeholder(dtype='float32', shape=[None, self.num_classes], name='input_y')
        self.total_shape = k.placeholder(dtype='int32', shape=[self.big_num + 1], name='total_shape')
        total_num = self.total_shape[-1]

        words_embedding = words_embedding_layer(self.input_words)
        pos1_embedding = pos1_embedding_layer(self.input_pos1)
        pos2_embedding = pos2_embedding_layer(self.input_pos2)
        # print(input_words.shape)
        # print(words_embedding.shape)
        # print(pos1_embedding.shape)
        # print(pos2_embedding.shape)
        # print(type(total_num))

        concat_embedding = k.concatenate(
            [words_embedding, pos1_embedding, pos2_embedding])  # shape: total_num, num_steps, d
        # print(concat_embedding.shape)

        output_h = BGRU_layer(concat_embedding)  # shape: total_num, num_steps, gru_size

        # word-level attention layer
        attention_w = tf.get_variable('attention_omega', [self.gru_size, 1])
        m = k.reshape(k.tanh(output_h), shape=[total_num * self.num_steps, self.gru_size])  # shape: total_num*num_steps, gru_size
        tmp = k.reshape(k.dot(m, attention_w), shape=[total_num, self.num_steps])
        alpha = k.reshape(k.softmax(tmp), shape=[total_num, 1, self.num_steps])
        r = k.batch_dot(alpha, output_h)
        attention_r = k.reshape(r, shape=[total_num, self.gru_size])

        # sentence-level attention
        sen_a = tf.get_variable('attention_A', [self.gru_size])
        sen_r = tf.get_variable('query_r', [self.gru_size, 1])
        relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.gru_size])
        sen_d = tf.get_variable('bias_d', [self.num_classes])
        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []
        for i in range(self.big_num):
            sen_repre.append(k.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]]))
            batch_size = self.total_shape[i + 1] - self.total_shape[i]
            e = k.reshape(k.dot(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])
            sen_alpha.append(k.reshape(k.softmax(e), [1, batch_size]))
            sen_s.append(k.reshape(k.dot(sen_alpha[i], sen_repre[i]), shape=(self.gru_size, 1)))
            sen_out.append(tf.add(k.reshape(k.dot(relation_embedding, sen_s[i]), shape=(self.num_classes,)), sen_d))

        # loss function
        self.total_loss = 0.0
        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []

        for i in range(self.big_num):
            self.prob.append(k.softmax(sen_out[i]))
            with k.name_scope('output'):
                self.predictions.append(k.argmax(self.prob[i], 0, ))
            with k.name_scope('loss'):
                self.loss.append(k.mean(k.categorical_crossentropy(self.input_y[i], sen_out[i], from_logits=True)))
                self.total_loss += self.loss[i]
            with k.name_scope('accuracy'):
                self.accuracy.append(k.cast(k.equal(self.predictions[i], k.argmax(self.input_y[i], 0)), 'float32'))

        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
