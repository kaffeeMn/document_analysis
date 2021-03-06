import numpy as np
from features import IdentityFeatureTransform
from collections import defaultdict
from operator import itemgetter

class CrossValidation(object):
    
    def __init__(self, category_bow_dict, n_folds):
        """Initialisiert die Kreuzvalidierung ueber gegebnen Daten
        
        Params:
            category_bow_dict: Dictionary, das fuer jede Klasse ein ndarray mit Merkmalsvektoren
                (zeilenweise) enthaelt.
            n_folds: Anzahl von Ausschnitten ueber die die Kreuzvalidierung berechnet werden soll.

        """
        self.__category_bow_list = list(category_bow_dict.iteritems())
        self.__n_folds = n_folds
        
    def validate(self, classifier, feature_transform=None):
        """Berechnet eine Kreuzvalidierung ueber die Daten,
        
        Params:
            classifier: Objekt, das die Funktionen estimate und classify implementieren muss.
            feature_transform: Objekt, das die Funktionen estimate und transform implementieren 
                muss. Optional: Falls None, wird keine Transformation durchgefuehrt.

        Returns:
            crossval_overall_result, crossval_class_results
            Resultate der Cross-Validierung ueber alle Daten und ueber Kategorien einzelnt
        """
        if feature_transform is None:
            # bei der identitaet wird "keine" Transformation vorgenommen
            feature_transform = IdentityFeatureTransform()
                
        crossval_overall_list = []
        crossval_class_dict = defaultdict(list)
        for fold_index in range(self.__n_folds):
            # Daten werden ab dem fold_index in training/ label Daten eingeteilt,
            # ab dem fold_index-ten document wird also jedes self.__n_folds-te
            # document in die test daten aufgenommen
            train_bow, train_labels, test_bow, test_labels = self.corpus_fold(fold_index)
            # (statischtisches) Schaetzen auf Basis der trainings-daten
            feature_transform.estimate(train_bow, train_labels)
            # Daten werden in einen neuen Raum transformiert
            train_feat = feature_transform.transform(train_bow)
            test_feat = feature_transform.transform(test_bow)
            # Mathematisches Modell schaetzt auf Basis der Trainingsdaten/
            # wird initialisiert
            classifier.estimate(train_feat, train_labels)
            estimated_test_labels = classifier.classify(test_feat)
            # evaluieren der Schaetzung
            classifier_eval = ClassificationEvaluator(estimated_test_labels, test_labels)
            # tuple als list abspeichern und danach in np.array convertieren?
            crossval_overall_list.append(list(classifier_eval.error_rate()))
            # Fehler je Klassen nach Klassen in unser dict fuer alle Fehlerraten abspeichern
            crossval_class_list = classifier_eval.category_error_rates()
            for category, err, n_wrong, n_samples in crossval_class_list:
                crossval_class_dict[category].append([err, n_wrong, n_samples])
        
        # [[error_rate, n_wrong, n_samlpes], ...]
        crossval_overall_mat = np.array(crossval_overall_list)
        # np.array von gewichteteten errorraten
        # Sum([error_rate * (n_samples / all samples considered), ... ])
        crossval_overall_result = CrossValidation.__crossval_results(crossval_overall_mat)

        # analog zu crossval_overall_result, jedoch mit sublisten nach Klassen sortiert
        # [(class, cval_res), ...]
        crossval_class_results = []
        for category in sorted(crossval_class_dict.keys()):
            crossval_class_mat = np.array(crossval_class_dict[category])
            crossval_class_result = CrossValidation.__crossval_results(crossval_class_mat)
            crossval_class_results.append((category, crossval_class_result))

        return crossval_overall_result, crossval_class_results
        
    @staticmethod
    def __crossval_results(crossval_mat):
        # Relative number of samples
        crossval_weights = crossval_mat[:, 2] / crossval_mat[:, 2].sum()
        # Weighted sum over recognition rates for all folds
        crossval_result = (crossval_mat[:, 0] * crossval_weights).sum()
        return crossval_result
        
        
        
    def corpus_fold(self, fold_index):
        """Berechnet eine Aufteilung der Daten in Training und Test
        
        Params:
            fold_index: Index des Ausschnitts der als Testdatensatz verwendet werden soll.
        
        Returns:
            Ergaenzen Sie die Dokumentation!
        """
        training_bow_mat = []
        training_label_mat = []
        test_bow_mat = []
        test_label_mat = []
        
        for category, bow_mat in self.__category_bow_list:
            # number of documents for the category
            n_category_samples = bow_mat.shape[0]
            
            # Select indices for fold_index-th test fold, remaining indices are used for training
            # last n-fold_index at every self.__n_folds steps
            test_indices = range(fold_index, n_category_samples, self.__n_folds)
            # first fold_index documents
            train_indices = [train_index for train_index in range(n_category_samples) 
                             if train_index not in test_indices]
            # matrix entries for training and test data
            category_train_bow = bow_mat[train_indices, :]
            category_test_bow = bow_mat[test_indices, :]
            # Construct label matrices ([x]*3 --> [x, x, x])
            # creating a corresponding list of categories (all are the same category, since we iterate
            # over the categories)
            category_train_labels = np.array([[category] * len(train_indices)])
            category_test_labels = np.array([[category] * len(test_indices)])

            # adding bow values
            training_bow_mat.append(category_train_bow)
            training_label_mat.append(category_train_labels.T)
            test_bow_mat.append(category_test_bow)
            test_label_mat.append(category_test_labels.T)

        training_bow_mat = np.vstack(tuple(training_bow_mat))
        training_label_mat = np.vstack(tuple(training_label_mat))
        test_bow_mat = np.vstack(tuple(test_bow_mat))
        test_label_mat = np.vstack(tuple(test_label_mat))

        return training_bow_mat, training_label_mat, test_bow_mat, test_label_mat



class ClassificationEvaluator(object):
    
    def __init__(self, estimated_labels, groundtruth_labels):
        """Initialisiert den Evaluator fuer ein Klassifikationsergebnis 
        auf Testdaten.
        
        Params:
            estimated_labels: ndarray (N x 1) mit durch den Klassifikator 
                bestimmten Labels.
            groundtruth_labels: ndarray (N x 1) mit den tatsaechlichen Labels.
                
        """
        self.__estimated_labels = estimated_labels
        self.__groundtruth_labels = groundtruth_labels
        # 
        # Bestimmen Sie hier die Uebereinstimmungen und Abweichungen der
        # durch den Klassifikator bestimmten Labels und der tatsaechlichen 
        # Labels
        
        self.__uneq = self.__estimated_labels != self.__groundtruth_labels
        self.__binary_result_mat = groundtruth_labels == estimated_labels

#     def error_rate(self, mask=None):
#         """Bestimmt die Fehlerrate auf den Testdaten.
#         
#         Params:
#             mask: Optionale boolsche Maske, mit der eine Untermenge der Testdaten
#                 ausgewertet werden kann. Nuetzlich fuer klassenspezifische Fehlerraten.
#                 Bei mask=None werden alle Testdaten ausgewertet.
#         Returns:
#             tuple: (error_rate, n_wrong, n_samlpes)
#             error_rate: Fehlerrate in Prozent
#             n_wrong: Anzahl falsch klassifizierter Testbeispiele
#             n_samples: Gesamtzahl von Testbeispielen
#         """
#         filtered = self.__uneq[mask]
#         n_wrong = np.sum(filtered.astype(int))
#         n_samples = len(filtered.reshape(-1))
#         return 100. * (float(n_wrong) / float(n_samples)), n_wrong, n_samples        
    def error_rate(self, mask=None):
        """Bestimmt die Fehlerrate auf den Testdaten.
        
        Params:
            mask: Optionale boolsche Maske, mit der eine Untermenge der Testdaten
                ausgewertet werden kann. Nuetzlich fuer klassenspezifische Fehlerraten.
                Bei mask=None werden alle Testdaten ausgewertet.
        Returns:
            tuple: (error_rate, n_wrong, n_samlpes)
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """

        if mask is None:
            mask = np.ones_like(self.__binary_result_mat, dtype=bool)
        masked_binary_result_mat = self.__binary_result_mat[mask]
        n_samples = len(masked_binary_result_mat)
        n_correct = masked_binary_result_mat.sum()
        n_wrong = n_samples - n_correct
        error_rate = n_wrong / float(n_samples)
        error_rate *= 100
        return error_rate, n_wrong, n_samples


    def category_error_rates(self):
        """Berechnet klassenspezifische Fehlerraten
        
        Returns:
            list von tuple: [ (category, error_rate, n_wrong, n_samlpes), ...]
            category: Label der Kategorie / Klasse
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        cats = sorted(set(self.__groundtruth_labels.reshape(-1)))
        return [(cat,) + self.error_rate(self.__groundtruth_labels == cat) for cat in cats]



