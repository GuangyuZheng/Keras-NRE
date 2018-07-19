import os
import numpy as np
import network
from utils import construct_data


path_prefix = os.getcwd()
word_embedding = np.load(os.path.join(path_prefix, 'data', 'vec.npy'))
train_y = np.load(os.path.join(path_prefix, 'data', 'small_y.npy'))
train_words = np.load(os.path.join(path_prefix, 'data', 'small_word.npy'))
train_pos1 = np.load(os.path.join(path_prefix, 'data', 'small_pos1.npy'))
train_pos2 = np.load(os.path.join(path_prefix, 'data', 'small_pos2.npy'))

settings = network.Settings()
settings.vocab_size = len(word_embedding)
settings.num_classes = len(train_y[0])
settings.sen_num = 20
settings.big_num = 50

model = network.BGRU_2ATT(word_embedding, settings).model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == "__main__":
    # model.summary()
    words, pos1, pos2, y = construct_data(settings.sen_num, train_words, train_pos1, train_pos2, train_y,
                                          len(word_embedding)-1, settings.num_steps)
    print('begin training:')
    model.fit([words, pos1, pos2], y, epochs=settings.num_epochs, batch_size=settings.big_num)

    save_path = os.path.join(path_prefix, 'model', 'my_model.h5')
    model.save(save_path)
    print('have saved model to ' + save_path)
