import bjp9pq_bag_of_words as bow
import numpy as np
import matplotlib.pyplot as plt
import os
import string


def perceptron(x_path, y_path, test_path, epochs = 6, title = 'Train Set'):
    feature_function, vocab, y_vect = bow.extract_feature_vector(x_path,y_path)
    vocab_size = len(vocab)
    y_range = 2
    acc = []
    ms = []

    for e in range(epochs):
        theta = np.random.random((y_range, feature_function.shape[0]))

        correct = 0
        m = np.zeros(theta.shape)
        for i in range(feature_function.shape[1]):
            #feature_vector, y = extract_vect(feature_function[:,i], vocab_size, i)
            out = np.dot(theta, feature_function[:,i])
            y_hat = list(out).index(max(out))
            y = y_vect[i]
            if y_hat != y:
                theta[y,:] += feature_function[:,i]
                theta[y_hat,:] -= feature_function[:,i]
            else:
                correct += 1
            m += theta

        m /= feature_function.shape[1]
        ms.append(m)
        acc.append(1- (float(correct) / feature_function.shape[1]))
        feature_function = feature_function.T
        np.random.shuffle(feature_function)
        feature_function = feature_function.T
        print("Epoch complete.")
    plt.clf()
    plt.title(title)
    plt.plot(np.arange(1, len(acc), 1), acc[1:])
    plt.show()


    #do prediction
    test_features = []
    with open(test_path, 'r') as f:
        a = 0
        for row in f:
            temp = [0 for i in range(len(vocab))]
            for word in row:
                word = word.lower()
                word = word.translate(str.maketrans({a: None for a in string.punctuation}))
                if word == '': continue
                if word in vocab: idx = vocab.index(word)
                else: idx = len(vocab)-1
                temp[idx] += 1
            for i in range(y_range-1): temp.extend(temp)
            test_features.append(temp)
    test_features = np.asarray(test_features).T
    y_hat =[]
    for i in range(test_features.shape[1]):
        out = np.dot(ms[-1], test_features[:, i])
        y_hat.append(list(out).index(max(out)))
    return y_hat


def main():
    trainx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trn.data')
    trainy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trn.label')
    devx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev.data')
    devy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev.label')
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tst.data')

    #testy = perceptron(trainx_path, trainy_path, test_path)
    testy = perceptron(devx_path, devy_path, test_path, title = 'Development Set')

    with open('bjp9pq-average-perceptron-test.pred', 'w') as f:
        for row in testy:
            f.write(str(row) + '\n')

    train, dev = bow.main()




if __name__  == '__main__':
    main()