import numpy as np

def viterbi_algorithm(line, vocabulary_list, pos_types, transition_matrix, emission_matrix):
    line = line.split(' ')
    K = len(pos_types) - 2
    M = len(line) + 1
    transition_matrix = np.asarray(transition_matrix)
    emission_matrix = np.asarray(emission_matrix)
    trellis_table = np.zeros((K, M))
    # trellis_table = [[0 for _ in range(len(vocabulary_list))] for _ in range(len(pos_types))]
    build_string = ''
    bp_path = []
    prev_word = ''
    max_prev_vm = 1
    bm = -1
    for m in range(M):  # loop over words
        max_vm_idx = -1
        end_case = m == M - 1
        if not end_case:
            split = line[m].split('/')
            word = split[0].strip('\n')
            orig_word = word
            if word not in vocabulary_list:
                word = 'UNK'
        for k in range(K):  # loop over pos_types
            if not end_case:
                vm = (max_prev_vm *
                      transition_matrix[bm+1, k] *
                      emission_matrix[k, vocabulary_list.index(word)])
            else:
                vm = (max_prev_vm * transition_matrix[bm+1, k])
            trellis_table[k, m] = vm
            #trellis_table[k, m, 1] = bm
            max_vm_idx = k if vm > trellis_table[max_vm_idx, m] else max_vm_idx
        bm = max_vm_idx
        prev_word = word
        max_prev_vm = trellis_table[max_vm_idx, m]
        build_string += orig_word + '/' + pos_types[bm + 1] + ' '
        bp_path.append(pos_types[bm+1])
    # print(build_string)
    return bp_path, build_string


def compare_pos(y, y_pred):
    same_count = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            same_count += 1
    #print('Line acc: ' + str(float(same_count) / len(y)))
    return same_count, len(y)


def get_viterbi(file_name, vocabulary_list, pos_types, transition_matrix, emission_matrix):
    acc_count = 0
    comp_count = 0
    write_file = ''
    with open(file_name, 'r') as f:
        for line in f:
            pred_pos, pred_string = viterbi_algorithm(line, vocabulary_list, pos_types, transition_matrix,
                                                      emission_matrix)
            write_file += pred_string[:-1] + '\n'
            pos = [w.split('/')[1] for w in line.split(' ')]
            same_pos_line_count, line_comp_count = compare_pos(pos, pred_pos)
            acc_count += same_pos_line_count
            comp_count += line_comp_count
            print('Cumm acc: ' + str(float(acc_count)/comp_count))

    with open('bjp9pq-viterbi-tuned.txt', 'w+') as f:
        f.write(write_file)
    return float(acc_count) / comp_count
