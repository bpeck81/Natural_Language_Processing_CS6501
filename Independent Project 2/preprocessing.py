import string
import viterbi as vit
import numpy as np

pos_types = ['Start', 'A', 'C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W', 'End']


def get_vocabulary(file_path, K=0):
    words_arr = []
    pos_arr = []
    vocab = {}
    with open(file_path, 'r') as f:
        for line in f:
            l = line.split(' ')
            pos_line = []
            word_line = []
            for w in l:
                split = w.split('/')
                word = split[0]
                pos = split[1]
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    vocab_list = list(vocab.keys())
    vocab['UNK'] = 0
    for key in vocab_list:
        if vocab[key] <= K:
            del vocab[key]
            vocab['UNK'] += 1
    print('Vocab Size: ' + str(len(vocab.keys())))
    print('UNK: ' + str(vocab['UNK']))
    return vocab


def write_transition_probability(trans_matrix, file_name):
    with open('bjp9pq' + file_name + '.txt', 'w+') as f:
        for i, row in enumerate(trans_matrix):
            for j, col in enumerate(row):
                line = (str(pos_types[i]) + ',' +
                        str(pos_types[j + 1]) + ',' + str(col) + '\n')  # +1 accounts for missing Start
                f.write(line)


def write_emission_probability(emission_matrix, vocab_list, file_name):
    with open('bjp9pq' + file_name + '.txt', 'w+') as f:
        for i, row in enumerate(emission_matrix):
            for j, col in enumerate(row):
                line = (str(pos_types[i + 1]) + ',' +
                        str(vocab_list[j]) + ',' + str(col) + '\n')
                f.write(line)


def get_write_transition_emission_probability(file_path, vocab, alpha=0.1, beta=0.1):
    vocab_list = list(vocab.keys())
    trans_matrix = [[0 for _ in range(len(pos_types) - 1)] for _ in range(len(pos_types) - 1)]  # exclude start or end
    emission_matrix = [[0 for _ in range(len(vocab_list))] for _ in
                       range(len(pos_types[1:-1]))]  # exclude start and end
    trans_count = 0
    emission_count = 0
    prev_pos_counts = [0 for _ in range(len(pos_types))]
    pos_counts = [0 for _ in range(len(pos_types))]
    with open(file_path, 'r') as f:
        line_count = 0
        for line in f:
            if line_count == 1000:
                pass
            line_count += 1
            l = line.split(' ')
            end_term = 'definitebrandonsend'
            l.append(end_term + '/End')
            prev_pos = 'Start'
            for i, w in enumerate(l):
                split = w.split('/')
                word = split[0]
                pos = split[1].strip('\n')
                if word not in vocab_list and word != end_term:
                    word = 'UNK'
                trans_matrix[pos_types.index(prev_pos)][pos_types.index(pos) - 1] += 1  # -1 orients for missing start
                trans_count += 1  # Note: Could be precomputed
                if word != end_term:
                    emission_matrix[pos_types.index(pos) - 1][
                        vocab_list.index(word)] += 1  # -1 orients for missing start
                    emission_count += 1  # Note: Could be precomputed
                pos_counts[pos_types.index(pos)] += 1
                prev_pos_counts[pos_types.index(prev_pos)] += 1
                prev_pos = pos

    trans_prob_matrix = [[float(col) / prev_pos_counts[i] for col in trans_matrix[i]] for i in range(len(trans_matrix))]
    emission_prob_matrix = [[float(col) / pos_counts[i+1] for col in emission_matrix[i]] for i in
                            range(len(emission_matrix))]
    smoothed_emission_prob_matrix = [
        [(float(col) + alpha) / (pos_counts[i + 1] + len(vocab_list) * alpha) for col in emission_matrix[i]]
        for i in range(len(emission_matrix))]
    # check below
    smoothed_trans_prob_matrix = [
        [(float(col) + beta) / float(prev_pos_counts[i] + (len(pos_types) - 2) * beta) for col in trans_matrix[i]]
        for i in range(len(trans_matrix))]
    write_transition_probability(trans_prob_matrix, '-tprob')
    write_emission_probability(emission_prob_matrix, vocab_list, '-eprob')
    write_transition_probability(smoothed_trans_prob_matrix, '-tprob-smoothed')
    write_emission_probability(smoothed_emission_prob_matrix, vocab_list, '-eprob-smoothed')
    write_transition_probability(trans_matrix, 'basetrans')
    write_emission_probability(emission_matrix, vocab_list, 'baseemiss')
    np.save('transition_prob_matrix', np.asarray(trans_prob_matrix))
    np.save('emission_prob_matrix', np.asarray(emission_prob_matrix))
    np.save('smoothed_transition_prob_matrix', np.asarray(smoothed_trans_prob_matrix))
    np.save('smoothed_emission_prob_matrix', np.asarray(smoothed_emission_prob_matrix))
    return trans_prob_matrix, emission_prob_matrix


def main():
    train_file = 'trn.pos'
    dev_file = 'dev.pos'
    test_file = 'test.pos'
    train_tweet_file = 'trn-tweet.pos'
    dev_tweet_file = 'dev-tweet.pos'
    train_vocab = get_vocabulary(train_file)
    transition_matrix, emission_matrix = get_write_transition_emission_probability(train_file, train_vocab)
    transition_matrix = np.load('smoothed_transition_prob_matrix.npy')
    emission_matrix = np.load('smoothed_emission_prob_matrix.npy')
    print(np.sum(transition_matrix, axis=1))
    print(np.sum(emission_matrix, axis=1))
    accuracy = vit.get_viterbi('dev.pos', list(train_vocab.keys()), pos_types, transition_matrix, emission_matrix)
    print('Net acc: ' + str(accuracy))


if __name__ == '__main__':
    main()
