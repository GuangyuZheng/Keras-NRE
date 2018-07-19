# padding data with blank and zero or truncate data
def construct_data(sen_num, words, pos1, pos2, blank, num_steps):
    train_words = []
    train_pos1 = []
    train_pos2 = []
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
        else:
            sentence_set = sentence_set[:sen_num]
            pos1_set = pos1_set[:sen_num]
            pos2_set = pos2_set[:sen_num]
        train_words.append(sentence_set)
        train_pos1.append(pos1_set)
        train_pos2.append(pos2_set)
    return train_words, train_pos1, train_pos2
