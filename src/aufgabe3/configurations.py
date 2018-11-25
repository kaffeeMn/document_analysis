import itertools
from corpus import CorpusLoader
from visualization import bar_plot

from evaluation import CrossValidation
from classification import KNNClassifier

from features import BagOfWords, WordListNormalizer, RelativeTermFrequencies, RelativeInverseDocumentWordFrequecies

from copy import deepcopy

class Configuration(object):
    '''
    '''


    def __init__(self, b_corpus, cat_word_dict, vocab,
                 voc_size, dist_metric, n_neighbours,  weighting='abs'):
        '''
        '''
        self.b_corpus = b_corpus
        self.__vocab = vocab[:voc_size]
        self.__metric = dist_metric
        self.__nn = n_neighbours
        self.__classifier = None
        self.__cross_validator = None
        
        
        # bow-matrices by category
        bow = BagOfWords(self.__vocab)
        self.category_bow_dict = bow.category_bow_dict(cat_word_dict)
        
        if weighting == 'abs':
            self.__wght = lambda x : x
        elif weighting == 'rel':
            self.__wght = RelativeTermFrequencies.weighting
        elif weighting == 'tfidf':
            tfidf = RelativeInverseDocumentWordFrequecies(self.__vocab, cat_word_dict)
            self.__wght = tfidf.weighting
        
    
    def fit(self):
        # initializing the classifier
        self.__classifier = KNNClassifier(self.__nn, self.__metric)
        
        
        # relative bow-matrices by category (weighted)
        rel_category_bow_dict = {cat : self.__wght(self.category_bow_dict[cat])
                                 for cat in self.category_bow_dict}
        
        # initializing the cross-validator (5 folds as in main)
        n_folds = 4
        self.__cross_validator = CrossValidation(category_bow_dict=rel_category_bow_dict, n_folds=n_folds)
        
    
    def eval(self):
        return self.__cross_validator.validate(self.__classifier)

class ExpConfigurations:
     
    def __init__(self, vocabulary, cat_wordlist_dict, voc_sizes=[], weightings=[], metrices=[], neighbours=[], ):
        self.__voc_szs = voc_sizes
        self.__wghts = weightings
        self.__mtrcs = metrices
        self.__nghbrs = neighbours
         
        self.__log = ["size", "weighting", "metric", "n_neighbours", "error"]
        self.__classifier = None
        self.__bow = None
         
        self.__vocab = vocabulary
        self.cat_wordlist_dict = cat_wordlist_dict
         
         
    def run(self):
        # all possible configurations
        for vs in self.__voc_szs:
            for wght in self.__wghts:
                for metric in self.__mtrcs:
                    for nn in self.__nghbrs:
                        # adding to the log
                        self.__classifier = KNNClassifier(nn, metric)
                        self.__log.append([vs, wght, metric, nn,
                                           self.eval(vs, wght)])
        # best result
        return self.best()
     
     
    def eval(self, vocab_size, weighting):
        # bow
        self.__bow = BagOfWords(self.__vocab[:vocab_size])
        category_bow_dict = self.__bow.category_bow_dict(self.cat_wordlist_dict)
        
        # weighting function
        if weighting == 'abs':
            wght = lambda x : x
        elif weighting == 'rel':
            wght = RelativeTermFrequencies.weighting
        elif weighting == 'tfidf':
            tfidf = RelativeInverseDocumentWordFrequecies(self.__vocab, self.cat_wordlist_dict)
            wght = tfidf.weighting
        
        
        wght_category_bow_dict = {cat : wght(category_bow_dict[cat])
                                 for cat in category_bow_dict}
        n_folds = 5
        
        c_val = CrossValidation(category_bow_dict=wght_category_bow_dict, n_folds=n_folds)
        return c_val.validate(self.__classifier)
    
    
    def best(self):
        res = self.__log[1:]
        if res:
            best = deepcopy(res[0])
            for r in res[1:]:
                if best[4][0] > r[4][0]:
                    best = deepcopy(r)
            return best
        return None