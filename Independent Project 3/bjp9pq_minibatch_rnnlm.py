
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

def load_data(file_name, batch_size):
    vocab = []
    x = [] # list of sentences
    y= [] # next word for each sentence of x
    exclude = set(['.',',','?','!','\n'])
    max_seq_len = 0
    with open(file_name) as f:
        for line in f:
            line = ''.join(ch for ch in line if ch not in exclude)
            tokens = line.split(" ")
            max_seq_len =  len(tokens) -1 if(len(tokens) -1) > max_seq_len else max_seq_len
            x.append(tokens[:-1]) # don't include <end> tag
            y.append(tokens[1:])
            vocab.extend([t for t in list(set(tokens)) if t not in vocab])

    #batch it up
    max_seq_len = 1000 # to remove disrepancies between train and dev data lengths
    vocab.insert(0,'<PAD>') #important that this is put at position zero to fill tensor with zeros
    counter = 0
    batch_x = []
    batch_y = []
    x_true = []
    y_true = []
    true_lens = []
    batch_true_lens = []
    for i in range(len(x)):
        if(counter == batch_size):
            counter = 0
            batch_x = [x for _,x in sorted(zip(batch_true_lens, batch_x), reverse=True)]
            batch_y = [y for _,y in sorted(zip(batch_true_lens, batch_y), reverse=True)]
            batch_true_lens = sorted(batch_true_lens, reverse=True)
            x_true.append(batch_x)
            y_true.append(batch_y)
            true_lens.append(batch_true_lens)
            batch_x = []
            batch_y = []
            batch_true_lens = []
        zero_count = max_seq_len - len(x[i]) # pad them
        batch_true_lens.append(len(x[i]))
        batch_x.append(x[i] + ['<PAD>' for _ in range(zero_count)])
        batch_y.append(y[i] + ['<PAD>' for _ in range(zero_count)])
        counter+=1
    return x_true, y_true, vocab, true_lens, max_seq_len

class lstm_lm(nn.Module):

    def __init__(self, hidden_dim, vocab_size, output_dim, embedding_dim = 32, num_layers=1, minibatch_size=1):
        super(lstm_lm, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim # assign num hidden units
        self.minibatch_size = minibatch_size
        #you have 32 dimensions to encode the vocab words
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.num_layers, batch_first=True) #create lstm
        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden() #assign hidden layers

    def init_hidden(self):
        # the axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim).cuda()))

    def forward(self, sentences, x_lens, max_seq_len):
        embeds = self.word_embeddings(sentences) #sentence is really the number corresponding to words
        X = torch.nn.utils.rnn.pack_padded_sequence(embeds, x_lens, batch_first=True)
        lstm_out, self.hidden = self.lstm(X, self.hidden)
        X, _  = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, total_length=max_seq_len ,batch_first=True)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        out_space = self.hidden2out(X)
        tag_scores = F.log_softmax(out_space, dim=1)
        return tag_scores
    def loss(self, y_hat, y, x, x_lens):
        y = y.view(-1)
        y_hat = y_hat.view(-1, self.output_dim)
        tag_pat_token = 0
        mask = (y> tag_pat_token).float()
        num_tokens =  int(torch.sum(mask))
        y_hat = y_hat[range(y_hat.shape[0]), y]
        y_hat  *= mask
        loss = -torch.sum(y_hat) / num_tokens
        return loss

def mini_batch_prepare_sequence(seq, vocab, batch_size):
    idxs = [[vocab.index(w) for w in seq[i]] for i in range(batch_size)]
    return torch.tensor(idxs, dtype=torch.long).cuda()

def train(x, y, vocab, batch_size, x_lens, max_seq_len):
    model = lstm_lm(32, len(vocab), len(vocab), 32, minibatch_size=batch_size).cuda()
    loss_function = nn.NLLLoss().cuda()
    cross_entropy = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=.1)
    # max_seq_len = max(x_lens)
    for epoch in range(1):
        losslst = []
        acclst = []
        for i in range(len(x)):
            sentences = x[i]
            tags = y[i]
            model.zero_grad()
            model.minibatch_size = batch_size
            model.hidden = model.init_hidden()
            sentence_in = mini_batch_prepare_sequence(sentences, vocab, batch_size)
            targets = mini_batch_prepare_sequence(tags, vocab, batch_size)
            tag_scores = model(sentence_in, x_lens[i], max_seq_len)
            # acc = (look_acc(tag_scores, tags, vocab, targets))
            # acclst.append(acc)
            loss = model.loss(tag_scores, targets, sentence_in, x_lens[i])
            losslst.append(float(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
        # print(float(sum(losslst))/len(losslst))
        # print(float(sum([a[0] for a in acclst]))/sum([a[1] for a in acclst]))
        print(epoch)
        print(sum(losslst))
        print(len(x))
        print(float(sum(losslst))/len(x))

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
    for batch_size in [16, 24,32,64]:
        # if batch_size != 16:
        trn_X, trn_Y, trn_vocab, x_lens,max_seq_len = load_data(trn_data_file, batch_size)
        ellapsed()

        print("training model")

        model, loss_func = train(trn_X, trn_Y, trn_vocab, batch_size, x_lens, max_seq_len)
        # save_model(model)
        # model, loss_func = load_model(trn_vocab)
        print('model loaded')
        ellapsed()
        # dev_X, dev_Y, dev_vocab, x_lens, max_seq_len = load_data(dev_data_file, batch_size)
        # model, loss_func = train(dev_X, dev_Y, dev_vocab, batch_size, x_lens, max_seq_len)

        # trn_perp = minibatch_perplexity(trn_vocab, trn_X, trn_Y, model,  batch_size,x_lens,max_seq_len)
        #
        # print("trn " + str(trn_perp))
        # ellapsed()
        # dev_perp = minibatch_perplexity(dev_vocab, dev_X, dev_Y, model , batch_size,x_lens,max_seq_len)
        #
        # print("dev " + str(dev_perp))
        ellapsed()

if __name__ == '__main__':
    main()