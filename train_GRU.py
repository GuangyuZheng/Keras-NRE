import os
import numpy as np
import network
from keras.models import load_model
from custom_layers import WordLevelAttentionLayer, SentenceLevelAttentionLayer


path_prefix = os.getcwd()
word_embedding = np.load(os.path.join(path_prefix, 'data', 'vec.npy'))
train_y = np.load(os.path.join(path_prefix, 'data', 'small_y.npy'))
train_words = np.load(os.path.join(path_prefix, 'data', 'small_word.npy'))
train_pos1 = np.load(os.path.join(path_prefix, 'data', 'small_pos1.npy'))
train_pos2 = np.load(os.path.join(path_prefix, 'data', 'small_pos2.npy'))

settings = network.Settings()
settings.vocab_size = len(word_embedding)
settings.num_classes = len(train_y[0])

BGRU_2ATT = network.BGRU_2ATT(word_embedding, settings)
model = BGRU_2ATT.model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == "__main__":
    # model.summary()

    print('begin training:')
    model.fit([train_words[:500], train_pos1[:500], train_pos2[:500]], train_y[:500], epochs=settings.num_epochs, batch_size=settings.big_num)

    save_path = os.path.join(path_prefix, 'model', 'my_model.h5')
    model.save_weights(save_path)
    print('have saved model to ' + save_path)
