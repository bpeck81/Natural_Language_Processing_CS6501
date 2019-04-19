## crf.py
## Author: CS 6501-005 NLP @ UVa
## Time-stamp: <yangfeng 10/14/2018 16:14:05>

from util import *
import sklearn_crfsuite as crfsuite
from sklearn_crfsuite import metrics

class CRF(object):
    def __init__(self, trnfile, devfile):
        self.trn_text = load_data(trnfile)
        self.dev_text = load_data(devfile)
        #
        print ("Extracting features on training data ...")
        self.trn_feats, self.trn_tags = self.build_features(self.trn_text)
        print ("Extracting features on dev data ...")
        self.dev_feats, self.dev_tags = self.build_features(self.dev_text)
        #
        self.model, self.labels = None, None

    def build_features(self, text):
        feats, tags = [], []
        for sent in text:
            N = len(sent.tokens)
            sent_feats = []
            for i in range(N):
                word_feats = self.get_word_features(sent, i)
                sent_feats.append(word_feats)
            feats.append(sent_feats)
            tags.append(sent.tags)
        return (feats, tags)

        
    def train(self):
        print ("Training CRF ...")
        self.model = crfsuite.CRF(
            algorithm='ap',
            max_iterations=5)
        self.model.fit(self.trn_feats, self.trn_tags)
        trn_tags_pred = self.model.predict(self.trn_feats)
        print('trn')
        self.eval(trn_tags_pred, self.trn_tags)
        dev_tags_pred = self.model.predict(self.dev_feats)
        print('dev')
        self.eval(dev_tags_pred, self.dev_tags)


    def eval(self, pred_tags, gold_tags):
        if self.model is None:
            raise ValueError("No trained model")
        print (self.model.classes_)
        print ("Acc =", metrics.flat_accuracy_score(pred_tags, gold_tags))

        
    def get_word_features(self, sent, i):
        """ Extract features with respect to time step i
        """
        # the i-th token
        word_feats = {'tok': sent.tokens[i]}
        # TODO for question 1
        word_feats['pos'] = sent.tags[i]
        # the i-th tag
        # 
        # TODO for question 2
        # Referenced https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html
        word_feats['first_letter'] = sent.tokens[i][0]
        word_feats['last_letter'] = sent.tokens[i][-1]
        #word_feats['first_two_letters'] = sent.tokens[i][:1] if len(sent.tokens[i]) > 1 else 'aa'
#        word_feats['last_two_letters'] = sent.tokens[i][:-2]
        word_feats['prev_tok'] = 'START_TOKEN' if i == 0 else sent.tokens[i-1]
        word_feats['next_tok'] = 'END_TOKEN' if i == len(sent.tokens) -1 else sent.tokens[i+1]
        word_feats['lower_tok'] = sent.tokens[i].lower()
#        word_feats['upper_tok'] = sent.tokens[i].isupper()
#        word_feats['digit_tok'] = sent.tokens[i].isdigit()
#        word_feats['title_tok'] = sent.tokens[i].istitle()


        # add more features here
        return word_feats


if __name__ == '__main__':
    trnfile = "trn-tweet.pos"
    devfile = "dev-tweet.pos"
    crf = CRF(trnfile, devfile)
    crf.train()

    
