import numpy as np
import datetime
import os
import network
from sklearn.metrics import average_precision_score
from keras.models import load_model


path_prefix = os.getcwd()
wordembedding = np.load(os.path.join(path_prefix, 'data', 'vec.npy'))
test_settings = network.Settings()
test_settings.vocab_size = 114044
test_settings.num_classes = 53
test_settings.big_num = 262 * 9

model = network.BGRU_2ATT(wordembedding, test_settings).model()

big_num_test = test_settings.big_num


# evaluate p@n
def eval_pn(test_y, test_word, test_pos1, test_pos2):
    allprob = []
    predict_result = model.predict([test_word, test_pos1, test_pos2], batch_size=32)

    for i in predict_result:
        allprob.append(i[1:])
    allprob = np.reshape(np.array(allprob), (-1))
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    order = np.argsort(-allprob)
    
    assert allans.shape == order.shape

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


if __name__ == '__main__':
    # ATTENTION: change pathname before you load your model
    pathname = os.path.join(path_prefix, 'model', 'weights.h5')
    model.load_weights(pathname)
    print("Evaluating P@N")

    print('Evaluating P@N for one')
    test_y = np.load(os.path.join(path_prefix, 'data', 'pone_test_y.npy'))
    test_word = np.load(os.path.join(path_prefix, 'data', 'pone_test_word.npy'))
    test_pos1 = np.load(os.path.join(path_prefix, 'data', 'pone_test_pos1.npy'))
    test_pos2 = np.load(os.path.join(path_prefix, 'data', 'pone_test_pos2.npy'))
    eval_pn(test_y, test_word, test_pos1, test_pos2)

    print('Evaluating P@N for two')
    test_y = np.load(os.path.join(path_prefix, 'data', 'ptwo_test_y.npy'))
    test_word = np.load(os.path.join(path_prefix, 'data', 'ptwo_test_word.npy'))
    test_pos1 = np.load(os.path.join(path_prefix, 'data', 'ptwo_test_pos1.npy'))
    test_pos2 = np.load(os.path.join(path_prefix, 'data', 'ptwo_test_pos2.npy'))
    eval_pn(test_y, test_word, test_pos1, test_pos2)

    print('Evaluating P@N for all')
    test_y = np.load(os.path.join(path_prefix, 'data', 'pall_test_y.npy'))
    test_word = np.load(os.path.join(path_prefix, 'data', 'pall_test_word.npy'))
    test_pos1 = np.load(os.path.join(path_prefix, 'data', 'pall_test_pos1.npy'))
    test_pos2 = np.load(os.path.join(path_prefix, 'data', 'pall_test_pos2.npy'))
    eval_pn(test_y, test_word, test_pos1, test_pos2)

    time_str = datetime.datetime.now().isoformat()
    print(time_str)
    print('Evaluating all test data and save data for PR curve')

    test_y = np.load(os.path.join(path_prefix, 'data', 'testall_y.npy'))
    test_word = np.load(os.path.join(path_prefix, 'data', 'testall_word.npy'))
    test_pos1 = np.load(os.path.join(path_prefix, 'data', 'testall_pos1.npy'))
    test_pos2 = np.load(os.path.join(path_prefix, 'data', 'testall_pos2.npy'))

    allprob = []
    predict_result = model.predict([test_word, test_pos1, test_pos2], batch_size=32)

    for i in predict_result:
        allprob.append(i[1:])
    allprob = np.reshape(np.array(allprob), (-1))
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    order = np.argsort(-allprob)

    assert allans.shape == order.shape

    print('saving all test result...')
    # ATTENTION: change the save path before you save your result !!
    np.save(os.path.join(path_prefix, 'out', 'allprob.npy'), allprob)
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

