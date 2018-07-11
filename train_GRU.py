import os
import numpy as np
import network
import tensorflow as tf
from keras import backend as k
import datetime

if __name__ == "__main__":
    path_prefix = os.getcwd()
    word_embedding = np.load(os.path.join(path_prefix, 'data', 'vec.npy'))
    train_y = np.load(os.path.join(path_prefix, 'data', 'small_y.npy'))
    train_words = np.load(os.path.join(path_prefix, 'data', 'small_word.npy'))
    train_pos1 = np.load(os.path.join(path_prefix, 'data', 'small_pos1.npy'))
    train_pos2 = np.load(os.path.join(path_prefix, 'data', 'small_pos2.npy'))

    settings = network.Settings()
    settings.vocab_size = len(word_embedding)
    settings.num_classes = len(train_y[0])

    big_num = settings.big_num

    sess = tf.Session()
    k.set_session(sess)

    with sess.as_default():
        model = network.BGRU_2ATT(word_embedding, settings)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(0.001)

        train_op = optimizer.minimize(model.final_loss, global_step=global_step)

        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver(max_to_keep=None)


        def train_one_step(word_batch, pos1_batch, pos2_batch, y_batch):
            feed_dict = {}
            total_word = []
            total_pos1 = []
            total_pos2 = []
            total_num = 0
            total_shape = []
            for i in range(len(word_batch)):
                total_shape.append(total_num)
                total_num += len(word_batch[i])
                for word in word_batch[i]:
                    total_word.append(word)
                for pos1 in pos1_batch[i]:
                    total_pos1.append(pos1)
                for pos2 in pos2_batch[i]:
                    total_pos2.append(pos2)
            total_shape.append(total_num)

            total_word = np.asarray(total_word)
            total_pos1 = np.asarray(total_pos1)
            total_pos2 = np.asarray(total_pos2)
            feed_dict[model.input_words] = total_word
            feed_dict[model.input_pos1] = total_pos1
            feed_dict[model.input_pos2] = total_pos2
            feed_dict[model.input_y] = y_batch
            feed_dict[model.total_shape] = total_shape

            temp, step, loss, accuracy, l2_loss, final_loss, predictions = sess.run(
                [train_op, global_step, model.total_loss, model.accuracy, model.l2_loss, model.final_loss,
                 model.predictions], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            accuracy = np.reshape(np.asarray(accuracy), (big_num,))
            acc = np.mean(accuracy)

            '''
            predictions = np.reshape(np.asarray(predictions), (big_num,))
            for i in range(big_num):
                print(str(predictions[i]) + " " + str(np.argmax(y_batch[i])))
            '''

            if step % 1 == 0:
                tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                print(tempstr)



        for one_epoch in range(settings.num_epochs):
            temp_order = list(range(len(train_words)))
            np.random.shuffle(temp_order)
            for i in range(int(len(temp_order) / float(settings.big_num))):
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

                train_one_step(temp_word, temp_pos1, temp_pos2, temp_y)