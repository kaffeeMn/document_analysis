import itertools
from corpus import CorpusLoader
from visualization import bar_plot

from evaluation import CrossValidation
from classification import KNNClassifier

from features import BagOfWords, WordListNormalizer, RelativeTermFrequencies, RelativeInverseDocumentWordFrequecies

from configurations import Configuration, ExpConfigurations
from copy import deepcopy

def aufgabe3():

    # ********************************** ACHTUNG **************************************
    # Die nun zu implementierenden Funktionen spielen eine zentrale Rolle im weiteren 
    # Verlauf des Fachprojekts. Achten Sie auf eine effiziente und 'saubere' Umsetzung. 
    # Verwenden Sie geeignete Datenstrukturen und passende Python Funktionen.
    # Wenn Ihnen Ihr Ansatz sehr aufwaendig vorkommt, haben Sie vermutlich nicht die
    # passenden Datenstrukturen / Algorithmen / (highlevel) Python / NumPy Funktionen
    # verwendet. Fragen Sie in diesem Fall!
    #
    # Schauen Sie sich jetzt schon gruendlich die Klassen und deren Interfaces in den
    # mitgelieferten Modulen an. Wenn Sie Ihre Datenstrukturen von Anfang an dazu 
    # passend waehlen, erleichtert dies deren spaetere Benutzung. Zusaetzlich bieten 
    # diese Klassen bereits etwas Inspiration fuer Python-typisches Design, wie zum 
    # Beispiel Duck-Typing.
    #
    # Zu einigen der vorgebenen Intefaces finden Sie Unit Tests in dem Paket 'test'. 
    # Diese sind sehr hilfreich um zu ueberpruefen, ob ihre Implementierung zusammen
    # mit anderen mitgelieferten Implementierungen / Interfaces funktionieren wird.
    # Stellen Sie immer sicher, dass die Unit tests fuer die von Ihnen verwendeten 
    # Funktionen erfolgreich sind. 
    # Hinweis: Im Verlauf des Fachprojekts werden die Unit Tests nach und nach erfolg-
    # reich sein. Falls es sie zu Beginn stoert, wenn einzelne Unit Tests fehlschlagen
    # koennen Sie diese durch einen 'decorator' vor der Methodendefinition voruebergehend
    # abschalten: @unittest.skip('')
    # https://docs.python.org/2/library/unittest.html#skipping-tests-and-expected-failures
    # Denken Sie aber daran sie spaeter wieder zu aktivieren.
    #
    # Wenn etwas unklar ist, fragen Sie!     
    # *********************************************************************************
    
    print('loading brown')
    CorpusLoader.load()
    brown = CorpusLoader.brown_corpus()
    
    # Um eine willkuerliche Aufteilung der Daten in Training und Test zu vermeiden,
    # (machen Sie sich bewusst warum das problematisch ist)
    # verwendet man zur Evaluierung von Klassifikatoren eine Kreuzvalidierung.
    # Dabei wird der gesamte Datensatz in k disjunkte Ausschnitte (Folds) aufgeteilt.
    # Jeder dieser Ausschnitte wird einmal als Test Datensatz verwendet, waehrend alle
    # anderen k-1 Ausschnitte als Trainings Datensatz verwendet werden. Man erhaehlt also
    # k Gesamtfehlerraten und k klassenspezifische Fehlerraten ide man jeweils zu einer
    # gemeinsamen Fehlerrate fuer die gesamte Kreuzvalidierung mittelt. Beachten Sie, 
    # dass dabei ein gewichtetes Mittel gebildet werden muss, da die einzelnen Test Folds
    # nicht unbedingt gleich gross sein muessen.

    # Fuehren Sie aufbauend auf den Ergebnissen aus aufgabe2 eine 5-Fold Kreuzvalidierung 
    # fuer den k-Naechste-Nachbarn Klassifikator auf dem Brown Corpus durch. Dazu koennen 
    # Sie die Klasse CrossValidation im evaluation Modul verwenden. 
    #
    # Vollziehen Sie dazu nach wie die Klasse die Daten in Trainging und Test Folds aufteilt.
    # Fertigen Sie zu dem Schema eine Skizze an. Diskutieren Sie Vorteile und Nachteile.
    # Schauen Sie sich an, wie die eigentliche Kreuzvalidierung funktioniert. Erklaeren Sie
    # wie das Prinzip des Duck-Typing hier angewendet wird.
    #
    # Hinweise: 
    #
    # Die Klasse CrossValidator verwendet die Klasse ClassificationEvaluator, die Sie schon
    # fuer aufgabe2 implementieren sollten. Kontrollieren Sie Ihre Umsetzung im Sinne der
    # Verwendung im CrossValidator.
    #
    # Fuer das Verstaendnis der Implementierung der Klasse CrossValidator ist der Eclipse-
    # Debugger sehr hilfreich.
    
#     brown_categories = brown.categories()
#     
#     n_neighbours = 1
#     metric = 'euclidean'
#     classifier = KNNClassifier(n_neighbours, metric)
#       
#     normalizer = WordListNormalizer()
#     normalized_words = normalizer.normalize_words(brown.words())
#     vocabulary = BagOfWords.most_freq_words(normalized_words[1], 500)
#     bow = BagOfWords(vocabulary)
#     cat_word_dict = {cat : [brown.words(doc) for doc in brown.fileids(categories=cat)] 
#                      for cat in brown_categories}
#       
#     n_folds = 5
#     category_bow_dict = bow.category_bow_dict(cat_word_dict)
#     cross_validator = CrossValidation(category_bow_dict=category_bow_dict, n_folds=n_folds)
#       
#     crossval_overall_result, crossval_class_results = cross_validator.validate(classifier)
#     print("ran cross validation for {}-nearest neighbour".format(n_neighbours))
#     print(crossval_overall_result)
#     print(crossval_class_results)

    # Bag-of-Words Weighting 
    #
    # Bisher enthalten die Bag-of-Words Histogramme absolute Frequenzen.
    # Dadurch sind die Repraesentationen abhaengig von der absoluten Anzahl
    # von Woertern in den Dokumenten.
    # Dies kann vermieden werden, in dem man die Bag-of-Words Histogramme mit
    # einem Normalisierungsfaktor gewichtet. 
    # 
    # Normalisieren Sie die Bag-of-Words Histogramme so, dass relative Frequenzen
    # verwendet werden. Implementieren und verwenden Sie die Klasse RelativeTermFrequencies 
    # im features Modul. 
    #
    # Wie erklaeren Sie das Ergebnis? Schauen Sie sich dazu noch einmal die 
    # mittelere Anzahl von Woertern pro Dokument an (aufgabe2).
    #
    # Wie in der Literatur ueblich, verwenden wir den
    # Begriff des "Term". Ein Term bezeichnet ein Wort aus dem Vokabular ueber
    # dem die Bag-of-Words Histogramme gebildet werden. Ein Bag-of-Words Histogramm
    # wird daher auch als Term-Vektor bezeichnet.
    
#     rel_category_bow_dict = {cat : RelativeTermFrequencies.weighting(category_bow_dict[cat])
#                              for cat in category_bow_dict}
#   
#     cross_validator = CrossValidation(category_bow_dict=rel_category_bow_dict, n_folds=n_folds)
#     crossval_overall_result, crossval_class_results = cross_validator.validate(classifier)
#     print("ran cross validation for {}-nearest neighbour (relative)".format(n_neighbours))
#     print(crossval_overall_result)
#     print(crossval_class_results)
    
    # Zusaetzlich kann man noch die inverse Frequenz von Dokumenten beruecksichtigen
    # in denen ein bestimmter Term vorkommt. Diese Normalisierung wird als  
    # inverse document frequency bezeichnet. Die Idee dahinter ist Woerter die in
    # vielen Dokumenten vorkommen weniger stark im Bag-of-Words Histogramm zu gewichten.
    # Die zugrundeliegende Annahme ist aehnlich wie bei den stopwords (aufgabe1), dass 
    # Woerter, die in vielen Dokumenten vorkommen, weniger Bedeutung fuer die 
    # Unterscheidung von Dokumenten in verschiedene Klassen / Kategorien haben als
    # Woerter, die nur in wenigen Dokumenten vorkommen. 
    # Diese Gewichtung laesst sich statistisch aus den Beispieldaten ermitteln.
    #
    # Zusammen mit der relativen Term Gewichtung ergibt sich die so genannte
    # "term frequency inverse document frequency"
    #
    #                            Anzahl von term in document                       Anzahl Dokumente
    # tfidf( term, document )  = ----------------------------   x   log ( ---------------------------------- ) 
    #                             Anzahl Woerter in document              Anzahl Dokumente die term enthalten
    #
    # http://www.tfidf.com
    #
    # Eklaeren Sie die Formel. Plotten Sie die inverse document frequency fuer jeden 
    # Term ueber dem Brown Corpus.   
    #
    # Implementieren und verwenden Sie die Klasse RelativeInverseDocumentWordFrequecies
    # im features Modul, in der Sie ein tfidf Gewichtungsschema umsetzen.
    # Ermitteln Sie die Gesamt- und klassenspezifischen Fehlerraten mit der Kreuzvalidierung.
    # Vergleichen Sie das Ergebnis mit der absolten und relativen Gewichtung.
    # Erklaeren Sie die Unterschiede in den klassenspezifischen Fehlerraten. Schauen Sie 
    # sich dazu die Verteilungen der Anzahl Woerter und Dokumente je Kategorie aus aufgabe1
    # an. In wie weit ist eine Interpretation moeglich? 
    
#     tfidf = RelativeInverseDocumentWordFrequecies(vocabulary, cat_word_dict)
#     rel_category_bow_dict = {cat : tfidf.weighting(category_bow_dict[cat])
#                              for cat in category_bow_dict}
#   
#     cross_validator = CrossValidation(category_bow_dict=rel_category_bow_dict, n_folds=n_folds)
#     crossval_overall_result, crossval_class_results = cross_validator.validate(classifier)
#     print("ran cross validation for {}-nearest neighbour (relative-inverse)".format(n_neighbours))
#     print(crossval_overall_result)
#     print(crossval_class_results)
      
    
    # Evaluieren Sie die beste Klassifikationsleistung   
    #
    # Ermitteln Sie nun die Parameter fuer die beste Klassifikationsleistung des 
    # k-naechste-Nachbarn Klassifikators auf dem Brown Corpus mit der Kreuzvalidierung.
    # Dabei wird gleichzeitig immer nur ein Parameter veraendert. Man hat eine lokal
    # optimale Parameterkonfiguration gefunden, wenn jede Aenderung eines Parameters
    # zu einer Verschlechterung der Fehlerrate fuehrt.
    #
    # Erlaeutern Sie warum eine solche Parameterkonfiguration lokal optimal ist.
    # 
    # Testen Sie mindestens die angegebenen Werte fuer die folgenden Parameter:
    # 1. Groesse des Vokabulars typischer Woerter (100, 500, 1000, 2000)
    # 2. Gewichtung der Bag-of-Words Histogramme (absolute, relative, relative with inverse document frequency)
    # 3. Distanzfunktion fuer die Bestimmung der naechsten Nachbarn (Cityblock, Euclidean, Cosine)
    # 4. Anzahl der betrachteten naechsten Nachbarn (1, 2, 3, 4, 5, 6)
    #
    # Erklaeren Sie den Effekt aller Parameter. 
    #
    # Erklaeren Sie den Effekt zwischen Gewichtungsschema und Distanzfunktion.
    
    # vocabulary sizes as specified
    voc_sizes = (100, 500, 1000, 2000)
    # weightings as specified
    weightings = ('abs','rel','tfidf')
    # metrices as specified
    dists = ("cityblock", "euclidean", "cosine")
    # numbers of neighbours as specified
    neighbours = (1, 2, 3, 4, 5, 6)
    
    # indices of best options, to keep track
    idx_vs = 0
    idx_wghts = 0
    idx_dsts = 0
    idx_nn = 0
    
    
    normalizer = WordListNormalizer()
    cat_word_dict = normalizer.category_wordlists_dict(corpus=brown)
    # Flatten the category word lists for computing overall word frequencies
    # The * operator expands the list/iterator to function arguments
    # itertools.chain concatenates all its parameters to a single list
    print 'Building Bag-of-Words vocabulary...'
    wordlists = itertools.chain(*(cat_word_dict.itervalues()))
    normalized_words = itertools.chain(*wordlists)
    vocabulary = BagOfWords.most_freq_words(normalized_words)
    
#     print("normalizing")
#     # normalizing
#     normalizer = WordListNormalizer()
#     # normalized words
#     normalized_words = normalizer.normalize_words(brown.words())[1]
#     print("normalized words")
#     # normalized wordlists
#     cat_word_dict = normalizer.category_wordlists_dict(brown)
#     print("normalized wordlists")
#     
#     # initializing the vocabulary at maximum size and then taking slices
#     vocabulary = BagOfWords.most_freq_words(normalized_words, 2000)
    
    
#     print('initializing configuration')
#     
#     exp_config = ExpConfigurations(vocabulary ,cat_word_dict,
#                                    voc_sizes, weightings, dists, neighbours)
#     print(exp_config.run())
#     for row in exp_config.__log:
#         print(row)
    
    
    config = Configuration(brown, cat_word_dict, vocabulary,
                           voc_sizes[idx_vs], dists[idx_dsts], neighbours[idx_nn], 
                           weightings[idx_wghts])
    config.fit()
    err_overall = config.eval()[0]
     
 
    # list of configuration results
    config_table = [['voc_sizes', 'metric', 'neighbours',  'weightings','err'],
                    [voc_sizes[idx_vs], dists[idx_dsts], neighbours[idx_nn], weightings[idx_wghts], err_overall]]
 
     
    print('calculating size')
    # local optimum for vocabulary size
    for idx, _ in enumerate(voc_sizes[1:]):
        tmp_config = Configuration(brown, cat_word_dict, vocabulary,
                                   voc_sizes[idx+1], dists[idx_dsts], neighbours[idx_nn], 
                                   weightings[idx_wghts])
        tmp_config.fit()
        tmp_err = tmp_config.eval()[0]
        if err_overall > tmp_err:
            err_overall = deepcopy(tmp_err)
            idx_vs = deepcopy(idx+1)
 
        config_table.append([ voc_sizes[idx+1], dists[idx_dsts], neighbours[idx_nn], 
                              weightings[idx_wghts],
                              tmp_err ])
    print(voc_sizes[idx_vs])
     
    print('calculating weights')
    # local optimum for weighting
    for idx, _ in enumerate(weightings[1:]):
        tmp_config = Configuration(brown, cat_word_dict, vocabulary,
                                   voc_sizes[idx_vs], dists[idx_dsts], neighbours[idx_nn], 
                                   weightings[idx+1])
        tmp_config.fit()
        tmp_err = tmp_config.eval()[0]
        if err_overall > tmp_err:
            err_overall = deepcopy(tmp_err)
            idx_wghts = deepcopy(idx+1)
         
        config_table.append([voc_sizes[idx_vs], dists[idx_dsts], neighbours[idx_nn], 
                              weightings[idx+1],
                              tmp_err ])
    print(weightings[idx_wghts])
     
    print('calculating metric')
    # local optimum for metric
    for idx, _ in enumerate(dists[1:]):
        tmp_config = Configuration(brown, cat_word_dict, vocabulary,
                                   voc_sizes[idx_vs], dists[idx+1], neighbours[idx_nn], 
                                   weightings[idx_wghts])
        tmp_config.fit()
        tmp_err = tmp_config.eval()[0]
        if err_overall > tmp_err:
            err_overall = deepcopy(tmp_err)
            idx_dsts = deepcopy(idx+1)
         
        config_table.append([voc_sizes[idx_vs], dists[idx+1], neighbours[idx_nn], 
                              weightings[idx_wghts],
                              tmp_err])
    print(dists[idx_dsts])
     
    print('calculating neighbours')
    # local optimum for neighbours
    for idx, _ in enumerate(neighbours[1:]):
        tmp_config = Configuration(brown, cat_word_dict, vocabulary,
                                   voc_sizes[idx_vs], dists[idx_dsts], neighbours[idx+1], 
                                   weightings[idx_wghts])
        tmp_config.fit()
        tmp_err = tmp_config.eval()[0]
        if err_overall > tmp_err:
            err_overall = deepcopy(tmp_err)
            idx_nn = deepcopy(idx+1)
         
        config_table.append([voc_sizes[idx_vs], dists[idx_dsts], neighbours[idx+1], 
                              weightings[idx_wghts],
                              tmp_err ])
    print(neighbours[idx_nn])
 
 
    # ideal should be 2000, relative, cityblock, 4
    print('local optimum at\nsize: {}\nweight: {}\nmetric: {}\nneighbours: {}'.format(voc_sizes[idx_vs],
                                                                                      weightings[idx_wghts],
                                                                                      dists[idx_dsts], 
                                                                                      neighbours[idx_nn]))
    for row in config_table:
        print(row)
    
if __name__ == '__main__':
    aufgabe3()
