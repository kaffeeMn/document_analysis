import numpy as np
import scipy.spatial.distance as distance
from features import BagOfWords , IdentityFeatureTransform




class KNNClassifier(object):
    
    def __init__(self, k_neighbors, metric):
        """Initialisiert den Klassifikator mit Meta-Parametern
        
        Params:
            k_neighbors: Anzahl der zu betrachtenden naechsten Nachbarn (int)
            metric: Zu verwendendes Distanzmass (string), siehe auch scipy Funktion cdist 
        """
        self.__k_neighbors = k_neighbors
        self.__metric = metric
        # Initialisierung der fuer Trainingsdaten als None. 
        self.__train_samples = None
        self.__train_labels = None

    def estimate(self, train_samples, train_labels):
        """Erstellt den k-Naechste-Nachbarn Klassfikator mittels Trainingdaten.
        
        Der Begriff des Trainings ist beim K-NN etwas irre fuehrend, da ja keine
        Modelparameter im eigentlichen Sinne statistisch geschaetzt werden. 
        Diskutieren Sie, was den K-NN stattdessen definiert.
        
        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing
        
        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        self.__train_samples = train_samples
        self.__train_labels = train_labels

    def classify(self, test_samples):
#         """Klassifiziert Test Daten.
#          
#         Params:
#             test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
#              
#         Returns:
#             test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
#          
#             mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
#         """
#         if self.__train_samples is None or self.__train_labels is None:
#             raise ValueError('Classifier has not been "estimated", yet!')
#  
#         # distances
#         dists = distance.cdist(test_samples, self.__train_samples, self.__metric)
#         # nearest k neighbours
#         nearest_k = np.argsort(dists, axis=1)[:,:self.__k_neighbors]
#         # labels of the idcs
#         k_labels = [self.__train_labels[n_k].reshape(-1) for n_k in nearest_k]
#          
#         # most frequent labels
#         return np.array([BagOfWords.most_freq_words(k_l, n_words=1) for k_l in k_labels])
#def classify(self, test_samples):
        """Klassifiziert Test Daten.
         
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
             
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
         
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        if self.__train_samples is None or self.__train_labels is None:
            raise ValueError('Classifier has not been "estimated", yet!')
 
         
        dist_mat = distance.cdist(test_samples, self.__train_samples,self.__metric)
        dist_mat_sort_ind = np.argsort(dist_mat, axis=1)
        dist_mat_neighbors_ind = dist_mat_sort_ind[:, :self.__k_neighbors]
        # dist_mat_neighbors might have more columns than self.__train_labels
        # result will be expanded along the third dimension and has to be merged
        # to a 2D array
        labels_mat_neighbors = self.__train_labels[dist_mat_neighbors_ind][:, :, 0]
        test_labels_list = []
        for labels_neighbors in labels_mat_neighbors:
            test_label = BagOfWords.most_freq_words(labels_neighbors.ravel(),n_words=1)[0]
            test_labels_list.append(test_label)
        test_labels = np.array([test_labels_list]).T
        return test_labels
        

class BayesClassifier(object):

    def __init__(self):
        """Initialisiert den Multinomial Bayes Klassifikator
        """
        # ndarray mit Klassen a-priori Wahrscheinlichkeiten
        self.__cat_apriori = None
        # ndarray mit Term-Kategorie Wahrscheinlichkeiten
        self.__term_cat_probs = None
        # ndarray mit allen bekannten Klassen (unabhaengig von Labels fuer Daten)
        self.__cat_labels = None
    
    def estimate(self, train_samples, train_labels):
        """Trainiert den Multinomial Bayes Klassfikator.
        
        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError('Implement me')
    
    def classify(self, test_samples):
        """Klassifiziert Test Daten.
        
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
        
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        if self.__cat_apriori is None or self.__term_cat_probs is None or \
                                            self.__cat_labels is None:
            raise ValueError('BayesClassifier has not been estimated!')
        
        raise NotImplementedError('Implement me')



