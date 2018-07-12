import tensorflow as tf
import numpy as np
import datetime
import os
import network
from sklearn.metrics import average_precision_score
from keras import backend as k

path_prefix = os.getcwd()

if __name__ == '__main__':
    # ATTENTION: change pathname before you load your model
    pathname = os.path.join(path_prefix, 'model', 'BGRU_2ATT_model-')

    wordembedding = np.load(os.path.join(path_prefix, 'data', 'vec.npy'))

    test_settings = network.Settings()
    test_settings.vocab_size = 114044
    test_settings.num_classes = 53
    test_settings.big_num = 262 * 9

    big_num_test = test_settings.big_num

    tf.reset_default_graph()
    sess = tf.Session()
    k.set_session(sess)
    k.set_learning_phase(0)
    with sess.as_default():

        def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

            feed_dict = {}
            total_shape = []
            total_num = 0
            total_word = []
            total_pos1 = []
            total_pos2 = []

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
            total_shape = np.array(total_shape)
            total_word = np.array(total_word)
            total_pos1 = np.array(total_pos1)
            total_pos2 = np.array(total_pos2)

            feed_dict[mtest.total_shape] = total_shape
            feed_dict[mtest.input_words] = total_word
            feed_dict[mtest.input_pos1] = total_pos1
            feed_dict[mtest.input_pos2] = total_pos2
            feed_dict[mtest.input_y] = y_batch

            loss, accuracy, prob = sess.run(
                [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
            return prob, accuracy

        # evaluate p@n
        def eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings):
            allprob = []
            acc = []
            for i in range(int(len(test_word) / float(test_settings.big_num))):
                prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                for single_prob in prob:
                    allprob.append(single_prob[1:])
            allprob = np.reshape(np.array(allprob), (-1))
            eval_y = []
            for i in test_y:
                eval_y.append(i[1:])
            allans = np.reshape(eval_y, (-1))
            order = np.argsort(-allprob)

            print('P@100:')
            top100 = order[:100]
            correct_num_100 = 0.0
            for i in top100:
                if allans[i] == 1:
                    correct_num_100 += 1.0
            print(correct_num_100 / 100)

            print('P@200:')
            top200 = order[:200]
            correct_num_200 = 0.0
            for i in top200:
                if allans[i] == 1:
                    correct_num_200 += 1.0
            print(correct_num_200 / 200)

            print('P@300:')
            top300 = order[:300]
            correct_num_300 = 0.0
            for i in top300:
                if allans[i] == 1:
                    correct_num_300 += 1.0
            print(correct_num_300 / 300)


        mtest = network.BGRU_2ATT(is_training=False, word_embeddings=wordembedding, settings=test_settings)

        saver = tf.train.Saver()

        # ATTENTION: change the list to the iters you want to test !!
        # testlist = range(9025,14000,25)
        testlist = [50]
        for model_iter in testlist:

            saver.restore(sess, pathname + str(model_iter))
            print(("Evaluating P@N for iter " + str(model_iter)))

            print('Evaluating P@N for one')
            test_y = np.load(os.path.join(path_prefix, 'data', 'pone_test_y.npy'))
            test_word = np.load(os.path.join(path_prefix, 'data', 'pone_test_word.npy'))
            test_pos1 = np.load(os.path.join(path_prefix, 'data', 'pone_test_pos1.npy'))
            test_pos2 = np.load(os.path.join(path_prefix, 'data', 'pone_test_pos2.npy'))
            eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings)

            print('Evaluating P@N for two')
            test_y = np.load(os.path.join(path_prefix, 'data', 'ptwo_test_y.npy'))
            test_word = np.load(os.path.join(path_prefix, 'data', 'ptwo_test_word.npy'))
            test_pos1 = np.load(os.path.join(path_prefix, 'data', 'ptwo_test_pos1.npy'))
            test_pos2 = np.load(os.path.join(path_prefix, 'data', 'ptwo_test_pos2.npy'))
            eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings)

            print('Evaluating P@N for all')
            test_y = np.load(os.path.join(path_prefix, 'data', 'pall_test_y.npy'))
            test_word = np.load(os.path.join(path_prefix, 'data', 'pall_test_word.npy'))
            test_pos1 = np.load(os.path.join(path_prefix, 'data', 'pall_test_pos1.npy'))
            test_pos2 = np.load(os.path.join(path_prefix, 'data', 'pall_test_pos2.npy'))
            eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings)

            time_str = datetime.datetime.now().isoformat()
            print(time_str)
            print('Evaluating all test data and save data for PR curve')

            test_y = np.load(os.path.join(path_prefix, 'data', 'testall_y.npy'))
            test_word = np.load(os.path.join(path_prefix, 'data', 'testall_word.npy'))
            test_pos1 = np.load(os.path.join(path_prefix, 'data', 'testall_pos1.npy'))
            test_pos2 = np.load(os.path.join(path_prefix, 'data', 'testall_pos2.npy'))
            allprob = []
            acc = []
            for i in range(int(len(test_word) / float(test_settings.big_num))):
                prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                for single_prob in prob:
                    allprob.append(single_prob[1:])
            allprob = np.reshape(np.array(allprob), (-1))
            order = np.argsort(-allprob)

            print('saving all test result...')
            current_step = model_iter

            # ATTENTION: change the save path before you save your result !!
            np.save(os.path.join(path_prefix, 'out', 'allprob_iter_' + str(current_step) + '.npy'), allprob)
            allans = np.load(os.path.join(path_prefix, 'data', 'allans.npy'))

            # caculate the pr curve area
            average_precision = average_precision_score(allans, allprob)
            print('PR curve area:' + str(average_precision))

            time_str = datetime.datetime.now().isoformat()
            print(time_str)
            print('P@N for all test data:')
            print('P@100:')
            top100 = order[:100]
            correct_num_100 = 0.0
            for i in top100:
                if allans[i] == 1:
                    correct_num_100 += 1.0
            print(correct_num_100 / 100)

            print('P@200:')
            top200 = order[:200]
            correct_num_200 = 0.0
            for i in top200:
                if allans[i] == 1:
                    correct_num_200 += 1.0
            print(correct_num_200 / 200)

            print('P@300:')
            top300 = order[:300]
            correct_num_300 = 0.0
            for i in top300:
                if allans[i] == 1:
                    correct_num_300 += 1.0
            print(correct_num_300 / 300)

