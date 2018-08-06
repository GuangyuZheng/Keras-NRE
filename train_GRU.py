import os
import numpy as np
import network
from keras.callbacks import ModelCheckpoint

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
    prev_weights_file = os.path.join(path_prefix, 'model', 'weights-{epoch:02d}.h5')
    weights_file = os.path.join(path_prefix, 'model', 'weights-{epoch:02d}.h5')

    try:
        model.load_weights(prev_weights_file)
        print("Continue training")
    except Exception:
        print("New model")

    checkpointer = ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1, save_best_only=False,
                                   save_weights_only=True, mode='auto')

    model.fit([train_words, train_pos1, train_pos2], train_y, batch_size=settings.big_num, epochs=settings.num_epochs,
              validation_split=0.05, initial_epoch=0, callbacks=[checkpointer])

