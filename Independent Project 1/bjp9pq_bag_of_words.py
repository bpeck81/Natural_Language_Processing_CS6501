import os
import numpy as np
import matplotlib.pyplot as plt
import string


def load_data():
    trainx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trn.data')
    trainy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trn.label')
    devx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev.data')
    devy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev.label')
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tst.data')
    train_features, _,_ = extract_feature_vector(trainx_path, trainy_path)
    dev_features,_,_ = extract_feature_vector(devx_path, devy_path)

    x_test = []
    with open(test_path, 'r') as f:
        for row in f:
            x_test.append(row.split(' '))
    test = np.asarray(x_test)
    return train_features, dev_features


def extract_feature_vector(trainx_path, trainy_path):
    y_train = []  # Y vector with 0 as negative sentence and 1 as positive sentence
    with open(trainy_path, 'r') as f:
        for val in f:
            y_train.append(int(val.strip('\n')))

    train_words = []  # 2D array rows = sentences cols = words in sentence
    train_dictionary_freq = {}  # Entire set dictonary mapping word to number of occurrences
    with open(trainx_path, 'r') as f:
        for row in f:
            temp = []
            for word in row.split(' '):
                word = word.lower()
                word = word.translate(str.maketrans({a: None for a in string.punctuation}))
                if word == '': continue
                if word in train_dictionary_freq:
                    train_dictionary_freq[word] += 1
                else:
                    train_dictionary_freq[word] = 1
                temp.append(word)
            train_words.append(temp)
    # Below is list of vocabulary ordered by frequency
    train_dictionary_lst = sorted(train_dictionary_freq.items(), key=lambda kv: kv[1])
    frequency_list = [val[1] for val in train_dictionary_lst]  # vocabulary ordered by freq
    vocabulary_list = [val[0] for val in train_dictionary_lst]  # word occurrence count ordered by freq
    linspace = np.arange(0, len(frequency_list), 1)
    #plt.xticks(linspace, vocabulary_list)
    plt.plot(linspace, frequency_list)
    plt.title("Words Vs Frequency")
    plt.xlabel('Words in Vocabulary')
    plt.ylabel('Frequency of Word')
    # plt.show()
    i = 0
    # Count words with single occurence
    frequency_list.append(0)
    oldvocabsize = len(vocabulary_list)

    while i < len(frequency_list) - 1 and frequency_list[i] <= 1:
        del train_dictionary_freq[vocabulary_list[i]]
        train_dictionary_lst.pop(0)  # Remove freq 1 from dict lst
        frequency_list[-1] += 1
        i += 1
    vocabulary_list = vocabulary_list[i:]
    vocabulary_list.append('UNK')
    frequency_list = frequency_list[i:]
    train_dictionary_lst.append(('UNK', frequency_list[-1]))
    print("Old vocabulary size " + str(oldvocabsize))
    print("Number of words with single occurrence: " + str(i))
    print("New vocabulary size " + str(oldvocabsize - i))
    print("Percentage of single occurrence words: " + str(100 * float(i) / oldvocabsize))
    linspace = np.arange(0, len(frequency_list), 1)
    plt.plot(linspace, frequency_list)
    plt.show()

    # Create feature vector with 2x number of words (one for pos y one for neg y)
    y_train_range = set(y_train)
    feature_vector = [[0 for i in range(len(vocabulary_list)) for j in range(len(y_train_range))]for k in range(len(train_words))]
    for i in range(len(train_words)):
        for j in range(len(train_words[i])):
            shift = y_train[i] * len(vocabulary_list)
            if train_words[i] in vocabulary_list:
                idx = vocabulary_list.index(train_words[i]) + shift
            else:
                idx = len(vocabulary_list) - 1 + shift
            feature_vector[i][idx] += 1
    feature_vector = np.asarray(feature_vector).T

    return feature_vector, vocabulary_list, y_train


def main():
    train, dev = load_data()
    return train, dev


if __name__ == '__main__':
    main()
