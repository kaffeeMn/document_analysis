import string
import numpy as np
from collections import defaultdict
from corpus import CorpusLoader
from nltk.stem.porter import PorterStemmer # IGNORE:import-error
from pygments.lexer import default
from ConfigParser import _default_dict

import operator


from nltk.corpus import stopwords

class AbsoluteTermFrequencies(object):
    """Klasse, die zur Durchfuehrung absoluter Gewichtung von Bag-of-Words
    Matrizen (Arrays) verwendet werden kann. Da Bag-of-Words zunaechst immer in
    absoluten Frequenzen gegeben sein muessen, ist die Implementierung dieser 
    Klasse trivial. Sie wird fuer die softwaretechnisch eleganten Unterstuetzung
    verschiedner Gewichtungsschemata benoetigt (-> Duck-Typing).   
    """
    @staticmethod
    def weighting(bow_mat):
        """Fuehrt die Gewichtung einer Bag-of-Words Matrix durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
        # Gibt das NumPy Array unveraendert zurueck, da die Bag-of-Words Frequenzen
        # bereits absolut sind.
        return bow_mat
    
    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'absolute'

    
class RelativeTermFrequencies(object):
    """Realisiert eine Transformation von in absoluten Frequenzen gegebenen 
    Bag-of-Words Matrizen (Arrays) in relative Frequenzen.
    """
    @staticmethod
    def weighting(bow_mat):
        """Fuehrt die relative Gewichtung einer Bag-of-Words Matrix (relativ im 
        Bezug auf Dokumente) durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
#         print((bow_mat.shape, np.array([[s] for s in np.sum(bow_mat, axis=1)]).shape))
        return bow_mat / np.array([np.sum(bow_mat, axis=1)]).T
    
    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'relative'

class RelativeInverseDocumentWordFrequecies(object):
    """Realisiert eine Transformation von in absoluten Frequenzen gegebenen 
    Bag-of-Words Matrizen (Arrays) in relative - inverse Dokument Frequenzen.
    """
    def __init__(self, vocabulary, category_wordlists_dict):
        """Initialisiert die Gewichtungsberechnung, indem die inversen Dokument
        Frequenzen aus dem Dokument Korpous bestimmt werden.
        
        Params:
            vocabulary: Python Liste von Woertern (das Vokabular fuer die 
                Bag-of-Words).
            category_wordlists_dict: Python dictionary, das zu jeder Klasse (category)
                eine Liste von Listen mit Woertern je Dokument enthaelt.
                Siehe Beschreibung des Parameters cat_word_dict in der Methode
                BagOfWords.category_bow_dict.
        """
        bow = BagOfWords(vocabulary)
        category_bow_dict = bow.category_bow_dict(category_wordlists_dict)
        
        # stacking documents
        keys = category_bow_dict.keys()
        doc_arr = category_bow_dict[keys[0]]
        for k in keys[1:]:
            doc_arr = np.vstack((doc_arr, category_bow_dict[k]))
        #print(doc_arr.shape)
        # count documents containing the term
        voc_doc_count = np.sum((doc_arr>0).astype(float), axis=0)
        #print(voc_doc_count.shape)
        n_docs = float(len(doc_arr))
        # Anzahl Dokumente / Anzahl Dokumente die den jeweiligen term enthalten 
        self.__inv_freq = np.log(n_docs / voc_doc_count)

    def weighting(self, bow_mat):
        """Fuehrt die Gewichtung einer Bag-of-Words Matrix durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
        return RelativeTermFrequencies.weighting(bow_mat) * self.__inv_freq
        

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'tf-idf'


# class BagOfWords(object):
#     """Berechnung von Bag-of-Words Repraesentationen aus Wortlisten bei 
#     gegebenem Vokabular.
#     """
#          
#     def __init__(self, vocabulary, term_weighting=AbsoluteTermFrequencies()):
#         """Initialisiert die Bag-of-Words Berechnung
#          
#         Params:
#             vocabulary: Python Liste von Woertern / Termen (das Bag-of-Words Vokabular).
#                 Die Reihenfolge der Woerter / Terme im Vokabular gibt die Reihenfolge
#                 der Terme in den Bag-of-Words Repraesentationen vor.
#             term_weighting: Objekt, das die weighting(bow_mat) Methode implemeniert.
#                 Optional, verwendet absolute Gewichtung als Default.
#         """
#         self.__vocabulary = vocabulary
#         self.__term_weighting = term_weighting
#          
#     def category_bow_dict(self, cat_word_dict):
#         """Erzeugt ein dictionary, welches fuer jede Klasse (category)
#         ein NumPy Array mit Bag-of-Words Repraesentationen enthaelt.
#          
#         Params:
#             cat_word_dict: Dictionary, welches fuer jede Klasse (category)
#                 eine Liste (Dokumente) von Listen (Woerter) enthaelt.
#                 cat : [ [word1, word2, ...],  <--  doc1
#                         [word1, word2, ...],  <--  doc2
#                         ...                         ...
#                         ]
#         Returns:
#             category_bow_mat: Ein dictionary mit Bag-of-Words Matrizen fuer jede
#                 Kategory. Eine Matrix enthaelt in jeder Zeile die Bag-of-Words 
#                 Repraesentation eines Dokuments der Kategorie. (d x t) bei d 
#                 Dokumenten und einer Vokabulargroesse t (Anzahl Terme). Die
#                 Reihenfolge der Terme ist durch die Reihenfolge der Worter / Terme
#                 im Vokabular (siehe __init__) vorgegeben.
#         """
#         return {cat : self.b_o_w_mat(cat_word_dict[cat]) for cat in cat_word_dict.keys()}
#                  
#     def b_o_w_mat(self, mat):
#         #bow_mat = np.array([[arr.count(voc) for voc in self.__vocabulary] for arr in mat]) 
#         bow_mat = np.zeros((len(mat), len(self.__vocabulary)))
#         ind_dct = {v:i for i,v in enumerate(self.__vocabulary)}
#         for i, doc in enumerate(mat):
#             for j, word in enumerate(doc):
#                 if word in self.__vocabulary:
#                     bow_mat[i,ind_dct[word]] += 1
#         return self.__term_weighting.weighting(bow_mat)
# #     

#     
class BagOfWords(object):
    """Berechnung von Bag-of-Words Repraesentationen aus Wortlisten bei 
    gegebenem Vokabular.
    """
         
    def __init__(self, vocabulary, term_weighting=AbsoluteTermFrequencies()):
        """Initialisiert die Bag-of-Words Berechnung
         
        Params:
            vocabulary: Python Liste von Woertern / Termen (das Bag-of-Words Vokabular).
                Die Reihenfolge der Woerter / Terme im Vokabular gibt die Reihenfolge
                der Terme in den Bag-of-Words Repraesentationen vor.
            term_weighting: Objekt, das die weighting(bow_mat) Methode implemeniert.
                Optional, verwendet absolute Gewichtung als Default.
        """
        self.__vocabulary = vocabulary
        self.__term_weighting = term_weighting
 
        self.__vocabulary_index_lut = {vword : index for index, vword in enumerate(vocabulary)}
         
         
    def category_bow_dict(self, cat_word_dict):
        """Erzeugt ein dictionary, welches fuer jede Klasse (category)
        ein NumPy Array mit Bag-of-Words Repraesentationen enthaelt.
         
        Params:
            cat_word_dict: Dictionary, welches fuer jede Klasse (category)
                eine Liste (Dokumente) von Listen (Woerter) enthaelt.
                cat : [ [word1, word2, ...],  <--  doc1
                        [word1, word2, ...],  <--  doc2
                        ...                         ...
                        ]
        Returns:
            category_bow_mat: Ein dictionary mit Bag-of-Words Matrizen fuer jede
                Kategory. Eine Matrix enthaelt in jeder Zeile die Bag-of-Words 
                Repraesentation eines Dokuments der Kategorie. (d x t) bei d 
                Dokumenten und einer Vokabulargroesse t (Anzahl Terme). Die
                Reihenfolge der Terme ist durch die Reihenfolge der Worter / Terme
                im Vokabular (siehe __init__) vorgegeben.
        """
     
        # ##################
        # Compute a Dictionary that contains
        # the document categories as keys and for each category
        # a matrix containing bag of words representations per document (row-wise) 
        # ##################
        cat_bow_dict = {}
        for category, document_words_list in cat_word_dict.iteritems():
             
            category_bow_mat = np.zeros((len(document_words_list),len(self.__vocabulary)))
 
            for index, document_words in enumerate(document_words_list):
                self.bow_histogram(document_words, category_bow_mat[index, :])
 
            category_bow_mat = self.__term_weighting.weighting(category_bow_mat)
 
            cat_bow_dict[category] = category_bow_mat
        return cat_bow_dict
         
                 
    
    def bow_histogram(self, word_list, bow_mat=None):
        #
        # Count words in word_list that also occur in the vocabulary.
        # Counts are accumulated in the numpy bow_mat array. The 
        # vocabulary_index_lut is a dictionary for fast vocabulary index
        # look-ups. If a word from word list is not part of the vocabulary
        # its occurrence is simply ignored (KeyError raised by the look-up
        # is ignored) 
        #
        if bow_mat is None:
            bow_mat = np.zeros((1, len(self.__vocabulary)))
         
        for word in word_list:
            try:
                vword_index = self.__vocabulary_index_lut[word]
                bow_mat[vword_index] += 1
            except KeyError:
                pass
        return bow_mat  
#   
       
    @staticmethod
    def most_freq_words(word_list, n_words=None):
        """Bestimmt die (n-)haeufigsten Woerter in einer Liste von Woertern.
         
        Params:
            word_list: Liste von Woertern
            n_words: (Optional) Anzahl von haeufigsten Woertern (top n). Falls
                n_words mit None belegt ist, sollen alle vorkommenden Woerter
                betrachtet werden.
             
        Returns:
            words_topn: Python Liste, die (top-n) am haeufigsten vorkommenden 
                Woerter enthaelt. Die Sortierung der Liste ist nach Haeufigkeit
                absteigend.
        """
        wcount_dict = defaultdict(int)
        for w in word_list:
            wcount_dict[w] += 1
             
        sorted_items = sorted(wcount_dict.items(), key=operator.itemgetter(1), reverse=True)
        words_topn = [i[0] for i in sorted_items]
         
        return words_topn if n_words is None else words_topn[:n_words]
  
    
class WordListNormalizer(object):
    
    def __init__(self, stoplist=None, stemmer=None):
        """Initialisiert die Filter
        
        Params: 
            stoplist: Python Liste von Woertern, die entfernt werden sollen
                (stopwords). Optional, verwendet NLTK stopwords falls None
            stemmer: Objekt, das die stem(word) Funktion implementiert. Optional,
                verwendet den Porter Stemmer falls None.
        """
        
        if stoplist is None:
            stoplist = CorpusLoader.stopwords_corpus()
        self.__stoplist = stoplist
        
        if stemmer is None:
            stemmer = PorterStemmer()
        self.__stemmer = stemmer
        self.__punctuation = string.punctuation
        self.__delimiters = ["''", '``', '--']
    
        
    def normalize_words(self, word_list):
        """Normalisiert die gegebenen Woerter nach in der Methode angwendeten
        Filter-Regeln (Gross-/Kleinschreibung, stopwords, Satzzeichen, 
        Bindestriche und Anfuehrungszeichen, Stemming)
        
        Params: 
            word_list: Python Liste von Worten.
            
        Returns:
            word_list_filtered, word_list_stemmed: Tuple von Listen
                Bei der ersten Liste wurden alle Filterregeln, bis auch stemming
                angewandt. Bei der zweiten Liste wurde zusaetzlich auch stemming
                angewandt.
        """
        lower_words = (str.lower(str(w)) for w in word_list)
        # filtering
        filtered_words = [w for w in lower_words
                          if w not in self.__stoplist
                          and w not in self.__punctuation
                          and w not in self.__delimiters]
        # additional stemming
        stemmed_words = [self.__stemmer.stem(w) for w in filtered_words]
        return filtered_words, stemmed_words
    
#     def cat_word_list_dict(self, brown):
#         # word by category
#         cat_word_dict_filtered = {}
#         cat_word_dict_stemmed = {}
#         for cat in brown.categories():
#             filtered, stemmed = self.normalize_words([brown.words(doc) 
#                                                       for doc in brown.fileids(categories=cat)])
#             cat_word_dict_filtered[cat] = filtered
#             cat_word_dict_stemmed[cat] = stemmed
#         
#         return cat_word_dict_filtered, cat_word_dict_stemmed


    def category_wordlists_dict(self, corpus):
        cat_word_dict = {}
        for category in corpus.categories():
            # getting documents
            category_documents = corpus.fileids(category)
            documents_words_list = []
            for document in category_documents:
                document_words = corpus.words(document)
                # normalizing word in document
                _, document_words_norm = self.normalize_words(document_words)
                documents_words_list.append(document_words_norm)
        
            cat_word_dict[category] = documents_words_list
        
        return cat_word_dict


class IdentityFeatureTransform(object):
    """Realisert eine Transformation auf die Identitaet, bei der alle Daten 
    auf sich selbst abgebildet werden. Die Klasse ist hilfreich fuer eine
    softwaretechnisch elegante Realisierung der Funktionalitaet "keine Transformation
    der Daten durchfuehren" (--> Duck-Typing).
    """
    def estimate(self, train_data, train_labels):
        pass
    def transform(self, data):
        return data

    
class TopicFeatureTransform(object):
    """Realsiert statistische Schaetzung eines Topic Raums und Transformation
    in diesen Topic Raum.
    """ 
    def __init__(self, topic_dim):
        """Initialisiert die Berechnung des Topic Raums
        
        Params:
            topic_dim: Groesse des Topic Raums, d.h. Anzahl der Dimensionen.
        """
        self.__topic_dim = topic_dim
        # Transformation muss in der estimate Methode definiert werden.
        self.__T = None
        self.__S_inv = None
        
    def estimate(self, train_data, train_labels): # IGNORE:unused-argument
        """Statistische Schaetzung des Topic Raums
        
        Params:
            train_data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
                Hinweis: Fuer den hier zu implementierenden Topic Raum werden die
                Klassenlabels nicht benoetigt. Sind sind Teil der Methodensignatur
                im Sinne einer konsitenten und vollstaendigen Verwaltung der zur
                Verfuegung stehenden Information.
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError('Implement me')
        
    def transform(self, data):
        """Transformiert Daten in den Topic Raum.
        
        Params:
            data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
        
        Returns:
            data_trans: ndarray der in den Topic Raum transformierten Daten 
                (d x topic_dim).
        """
        raise NotImplementedError('Implement me')

