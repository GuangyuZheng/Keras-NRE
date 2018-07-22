import numpy as np
import random


# padding data with blank and zero or truncate data
def construct_data(sen_num, words, pos1, pos2, blank, num_steps):
    train_words = []
    train_pos1 = []
    train_pos2 = []
    if isinstance(words, type(np.array([0]))):
        words = words.tolist()
    if isinstance(pos1, type(np.array([0]))):
        pos1 = pos1.tolist()
    if isinstance(pos2, type(np.array([0]))):
        pos2 = pos2.tolist()

    for i in range(len(words)):
        sentence_set = words[i]
        pos1_set = pos1[i]
        pos2_set = pos2[i]
        if len(sentence_set) < sen_num:
            j = len(sentence_set)
            while j < sen_num:
                blank_sentence = [blank for x in range(num_steps)]
                zero_pos1 = [0 for x in range(num_steps)]
                zero_pos2 = [0 for x in range(num_steps)]
                sentence_set.append(blank_sentence)
                pos1_set.append(zero_pos1)
                pos2_set.append(zero_pos2)
                j += 1
            train_words.append(sentence_set)
            train_pos1.append(pos1_set)
            train_pos2.append(pos2_set)
        else:
            index = list(range(len(sentence_set)))
            sample_index = random.sample(index, sen_num)
            sample_sentence_set = []
            sample_pos1_set = []
            sample_pos2_set = []
            for idx in sample_index:
                sample_sentence_set.append(sentence_set[idx])
                sample_pos1_set.append(pos1_set[idx])
                sample_pos2_set.append(pos2_set[idx])
            train_words.append(sample_sentence_set)
            train_pos1.append(sample_pos1_set)
            train_pos2.append(sample_pos2_set)
    return train_words, train_pos1, train_pos2
