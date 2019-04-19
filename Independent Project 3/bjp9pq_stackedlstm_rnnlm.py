
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bjp9pq_perplexity import *
import string
import time
import numpy as np


torch.cuda.manual_seed(1)
start_time = time.time()

def load_data(file_name):
    vocab = []
    x = [] # list of sentences
    y= [] # next word for each sentence of x
    exclude = set(['.',',','?','!','\n'])
    with open(file_name) as f:
        for line in f:
            line = ''.join(ch for ch in line if ch not in exclude)
            tokens = line.split(" ")
            x.append(tokens[:-1]) # don't include <end> tag
            y.append(tokens[1:])
            vocab.extend([t for t in list(set(tokens)) if t not in vocab])
    #x is list of sentences
    return x, y, vocab

class lstm_lm(nn.Module):

    def __init__(self, hidden_dim, vocab_size, output_dim, embedding_dim = 32, num_layers=1, minibatch_size=1):
        super(lstm_lm, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim # assign num hidden units
        self.minibatch_size = minibatch_size
        #you have 32 dimensions to encode the vocab words
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.num_layers) #create lstm
        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden() #assign hidden layers

    def init_hidden(self):
        # the axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) #sentence is really the number corresponding to words
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), self.minibatch_size, -1), self.hidden)
        out_space = self.hidden2out(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(out_space, dim=1) #question??
        return tag_scores

def prepare_sequence(seq, vocab):
    idxs = [vocab.index(w) for w in seq]
    return torch.tensor(idxs, dtype=torch.long).cuda()

def train(x, y, vocab, num_layers):
    model = lstm_lm(32, len(vocab), len(vocab), 32, num_layers=num_layers).cuda()
    loss_function = nn.NLLLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=.1)
    for epoch in range(1):
        losslst = []
        acclst = []
        for i in range(len(x)):
            sentence = x[i]
            tags = y[i]
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = prepare_sequence(sentence, vocab)
            targets = prepare_sequence(tags, vocab)
            tag_scores = model(sentence_in)
            # acc = (look_acc(tag_scores, tags, vocab, targets))
            # acclst.append(acc)
            loss = loss_function(tag_scores, targets)
            # losslst.append(float(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
        # print(float(sum(losslst))/len(losslst))
        # print(float(sum([a[0] for a in acclst]))/sum([a[1] for a in acclst]))
        print(epoch)

    return model, loss_function

def look_acc(tag_scores, tags, vocab, targets):
    best = tag_scores.data.max(1)[1]
    acc = float(sum([1 if best[i] == targets[i] else 0 for i in range(len(best))]))
    return (acc, len(best))


def save_model(model):
    torch.save(model.state_dict(), 'simplemodel.model')

def load_model(vocab, path = 'simplemodel.model'):
    model = lstm_lm(32, len(vocab), len(vocab), 32)
    model.load_state_dict(torch.load(path))
    loss_function = nn.NLLLoss()
    return model, loss_function

def ellapsed():
    print("------%s mins ------" %((time.time() - start_time)/float(60)))

def main():
    trn_data_file = 'trn-wiki.txt'
    dev_data_file = 'dev-wiki.txt'
    tst_data_file = 'tst-wiki.txt'

    print("loading data")
    trn_X, trn_Y, trn_vocab = load_data(trn_data_file)
    ellapsed()

    print("training model")

    for layer_num in [2,3]:
        model, loss_func = train(trn_X, trn_Y, trn_vocab, layer_num)
        # save_model(model)

        # model, loss_func = load_model(trn_vocab)
        print('model loaded')
        ellapsed()

        trn_perp = calc_perplexity(trn_vocab, trn_X, trn_Y, model, loss_func)

        print("trn " + str(trn_perp))
        ellapsed()
        dev_X, dev_Y, dev_vocab = load_data(dev_data_file)
        dev_perp = calc_perplexity_2(dev_vocab, dev_X, dev_Y, model)

        print("dev " + str(dev_perp))
        ellapsed()

        # trn_perp = calc_perplexity_2(trn_vocab, trn_X, trn_Y, model)
        # print("perp2 " + str(trn_perp))

if __name__ == '__main__':
    main()