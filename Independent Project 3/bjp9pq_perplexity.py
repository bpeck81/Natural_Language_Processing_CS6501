import math
from bjp9pq_minibatch_rnnlm import mini_batch_prepare_sequence
import bjp9pq_simple_rnnlm
import torch
from bjp9pq_simple_rnnlm import prepare_sequence
def calc_perplexity(vocab, X, Y, model, loss_function, batch_size = 1, x_lens = None, max_seq_lens = -1):
    model.eval()
    avg_loss = 0.0
    counts = 0.1
    with torch.no_grad():
        for i in range(len(X)):
            sent = X[i]
            pred_seq = Y[i]
            inputs = prepare_sequence(sent, vocab )
            targets = prepare_sequence(pred_seq, vocab )
            model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
            try:
                tag_scores = model(inputs)
                loss = loss_function(tag_scores, targets)
                avg_loss += loss
                counts +=1
            except Exception as e:
                pass
        try:
            return torch.exp(avg_loss/float(counts))
        except:
            return math.exp(avg_loss/float(counts))


def minibatch_perplexity(vocab, X, Y, model, batch_size, x_lens, max_seq_lens):
    model.eval()
    avg_loss = 0.0
    counts = 0.1
    with torch.no_grad():
        for i in range(len(X)):
            sent = X[i]
            pred_seq = Y[i]
            sentence_in = mini_batch_prepare_sequence(sent, vocab, batch_size)
            targets = mini_batch_prepare_sequence(pred_seq, vocab, batch_size)
            model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
            try:
                tag_scores = model(sentence_in, x_lens, max_seq_lens)
                loss = model.loss(tag_scores, targets, sentence_in, x_lens[i])
                avg_loss += loss
                counts +=1
            except Exception as e:
                pass
        try:
            return torch.exp(avg_loss/float(counts))
        except:
            return math.exp(avg_loss/float(counts))

#This works but is slower than the above though equivalent
def calc_perplexity_2(vocab, X, Y, model, batch_size = 1):
    model.eval()
    N = 0
    total = 0
    sums = []
    Ns = []
    with torch.no_grad():
        for i in range(len(X)):
            sent = X[i]
            pred_seq = Y[i]
            try:
                inputs = prepare_sequence(sent, vocab, batch_size)
                targets = prepare_sequence(pred_seq, vocab, batch_size)
                model.hidden = model.init_hidden()
                log_probs = model(inputs)
                probs = []
                cur_prob = [log_probs[j, int(targets[j])] for j in range(len(targets))]
                total += sum(cur_prob)
                N += len(targets)
            except Exception as E:
                pass
                # print(E)
                #print("missed "+str(i))
        avg = float(total) / float(N)
        print(avg)
        try:
            return torch.exp(-1*avg)
        except:
            return math.exp(-1*avg)
