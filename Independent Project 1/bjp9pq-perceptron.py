import bjp9pq_bag_of_words as bow
import numpy as np
import matplotlib.pyplot as plt
import os


def perceptron(x_path, y_path, epochs = 6, title='Train Set'):
    feature_function, vocab, y_vect = bow.extract_feature_vector(x_path,y_path)
    vocab_size = len(vocab)
    y_range = 2
    acc = []
    for e in range(epochs):
        theta = np.random.random((y_range, feature_function.shape[0]))
        correct = 0
        for i in range(feature_function.shape[1]):
            out = np.dot(theta, feature_function[:,i])
            y_hat = list(out).index(max(out))
            y = y_vect[i]
            if y_hat != y:
                theta[y,:] += feature_function[:,i]
                theta[y_hat,:] -= feature_function[:,i]
            else:
                correct += 1
        acc.append(1- (float(correct) / feature_function.shape[1]))
        feature_function = feature_function.T
        np.random.shuffle(feature_function)
        feature_function = feature_function.T
        print("Epoch complete.")
    plt.clf()
    plt.title(title)
    plt.plot(np.arange(1, len(acc), 1), acc[1:])
    plt.show()
    pass





def main():
    trainx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trn.data')
    trainy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trn.label')
    devx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev.data')
    devy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev.label')
    perceptron(trainx_path, trainy_path, title = 'Train Set')
    perceptron(devx_path, devy_path, title = 'Development Set')
    train, dev = bow.main()




if __name__  == '__main__':
    main()