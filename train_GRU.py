import os
import numpy as np
import network

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
# previous_weight = os.path.join(path_prefix, 'model', 'my_model.h5')
# model.load_weights(previous_weight)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == "__main__":
    # model.summary()

    print('begin training:')

    current_step = 0
    for one_epoch in range(settings.num_epochs):
        temp_order = list(range(len(train_words)))
        np.random.shuffle(temp_order)
        for i in range(int(len(temp_order)/float(settings.big_num))):
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            temp_y = []
            temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]

            for k in temp_input:
                temp_word.append(train_words[k])
                temp_pos1.append(train_pos1[k])
                temp_pos2.append(train_pos2[k])
                temp_y.append(train_y[k])
            num = 0
            for single_word in temp_word:
                num += len(single_word)

            if num > 1500:
                print('out of range')
                continue

            temp_word = np.array(temp_word)
            temp_pos1 = np.array(temp_pos1)
            temp_pos2 = np.array(temp_pos2)
            temp_y = np.array(temp_y)

            value = model.train_on_batch([temp_word, temp_pos1, temp_pos2], temp_y)

            current_step += settings.big_num

            if current_step > 9000 and current_step % 500 == 0:
            # if current_step == 50:
                print('step: ' + str(current_step) + ' loss:' + str(value[0]) + ' acc: ' + str(value[1]))
                save_path = os.path.join(path_prefix, 'model', 'my_model-'+str(current_step)+'.h5')
                model.save_weights(save_path)
                print('have saved model to ' + save_path)

