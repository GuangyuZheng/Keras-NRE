from keras import backend as k
from keras.models import Model
from keras.layers import Input, Embedding, GRU, Bidirectional, concatenate, Lambda, LSTM
from custom_layers import WordLevelAttentionLayer, SentenceLevelAttentionLayer
from keras import regularizers


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
        self.sen_num = 10  # sentence number for relation extraction
        # the number of entity pairs of each batch during training or testing
        self.big_num = 50
        # penalty for regularizer
        self.penalty_rate = 0.0001


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
        self.sen_num = settings.sen_num
        self.rate = settings.penalty_rate

    def flatten(self, x):
        return k.reshape(x, shape=(-1, self.num_steps))

    def reshape(self, x):
        return k.reshape(x, shape=(-1, self.sen_num, self.gru_size))

    def model(self):
        words_embedding_layer = Embedding(len(self.word_embeddings), len(self.word_embeddings[0]),
                                          weights=[self.word_embeddings], trainable=True,
                                          embeddings_regularizer=regularizers.l2(self.rate),
                                          activity_regularizer=regularizers.l2(self.rate))

        pos1_embedding_layer = Embedding(self.pos_num, self.pos_size, embeddings_initializer='glorot_uniform',
                                         trainable=True, embeddings_regularizer=regularizers.l2(self.rate),
                                         activity_regularizer=regularizers.l2(self.rate))

        pos2_embedding_layer = Embedding(self.pos_num, self.pos_size, embeddings_initializer='glorot_uniform',
                                         trainable=True, embeddings_regularizer=regularizers.l2(self.rate),
                                         activity_regularizer=regularizers.l2(self.rate))

        BGRU_layer = Bidirectional(GRU(units=self.gru_size, return_sequences=True, dropout=1 - self.keep_prob,
                                       kernel_regularizer=regularizers.l2(self.rate),
                                       bias_regularizer=regularizers.l2(self.rate),
                                       activity_regularizer=regularizers.l2(self.rate)), merge_mode='sum')

        flatten_layer = Lambda(self.flatten, name='flatten')
        reshape_layer = Lambda(self.reshape, name='reshape')

        input_words = Input(shape=(self.sen_num, self.num_steps), name='input_words')
        input_pos1 = Input(shape=(self.sen_num, self.num_steps), name='input_pos1')
        input_pos2 = Input(shape=(self.sen_num, self.num_steps), name='input_pos2')
        # self.input_y = Input(shape=(self.num_classes,), name='input_y')

        input_words_flatten = flatten_layer(input_words)
        input_pos1_flatten = flatten_layer(input_pos1)
        input_pos2_flatten = flatten_layer(input_pos2)

        words_embedding = words_embedding_layer(input_words_flatten)
        pos1_embedding = pos1_embedding_layer(input_pos1_flatten)
        pos2_embedding = pos2_embedding_layer(input_pos2_flatten)
        concat_embedding = concatenate([words_embedding, pos1_embedding, pos2_embedding])
        # print(concat_embedding.shape)

        output_h = BGRU_layer(concat_embedding)
        # print(output_h.shape)

        attention_r = WordLevelAttentionLayer()(output_h)

        attention_r = reshape_layer(attention_r)
        # print(attention_r.shape)

        sen_out = SentenceLevelAttentionLayer(self.big_num, self.num_classes)(attention_r)

        # print('sen out shape:')
        # print(sen_out.shape)

        model = Model(inputs=[input_words, input_pos1, input_pos2], outputs=sen_out)

        return model







