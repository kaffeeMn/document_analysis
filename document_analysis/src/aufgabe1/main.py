
import datetime
import string
import nltk
from nltk.stem import PorterStemmer # IGNORE:import-error
from corpus import CorpusLoader
from features import WordListNormalizer, BagOfWords
from visualization import hbar_plot

import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import stopwords

def aufgabe1():
    
    #
    # In der ersten Aufgabe sollen Sie sich mit dem Brown Corpus 
    # vertraut machen. 
    #  - Laden Sie den Corpus und schauen Sie sich dessen Aufbau an.
    #  - Analysieren Sie den Corpus in dem Sie Wortstatistiken bestimmen.
    #  - Verbessern Sie die Aussagekraft der Statistiken.
     
    # Laden des Corpus
    # Fuer das Fachprojekt benoetigen Sie die NLTK (http://www.nltk.org/)
    # Datensaetze "brown" und "stopwords". Falls diese noch nicht lokal 
    # auf Ihrem Rechner verfuegbar sein sollten, koennen Sie sie ueber
    # den "NLTK Downloader" herunterladen. Ein entsprechender Dialog
    # oeffnet sich in diesem Fall automatisch.   
    CorpusLoader.load()
    
    #
    # Im Folgenden werden einige grundlegende Statistiken des Brown Corpus
    # ausgegeben, die vor allem etwas ueber dessen Struktur / Aufbau
    # aussagen.
    # Siehe auch: http://en.wikipedia.org/wiki/Brown_Corpus
    #
    # Der Corpus enthaelt verschiedene Kategorien, in die Dokumente
    # einsortiert sind. Ein Dokument besteht aus Woertern.
    # Als naechstes sehen Sie, wie Sie auf Kategorien, Dokumente und
    # Woerter zugreifen koennen.
    brown = CorpusLoader.brown_corpus()
    brown_categories = brown.categories()
    brown_documents = brown.fileids()
    brown_words = brown.words()
    
    # Geben Sie nun die Gesamtanzahl von Kategorien, Dokumenten und Woertern
    # mit print auf der Konsole aus.  
    #print(type(brown_categories),type(brown_documents),type(brown_words))
    print('Anzahlen:\nKategorien {}\nDokumente {}\nWoerter {}'.format(len(brown_categories),
                                                                      len(brown_documents),
                                                                      len(brown_words)))

    # Geben Sie die Namen der einzelnen Kategorien aus. 
    print(brown_categories)
     
    # Bisher haben Sie noch keine Information ueber die Struktur des Brown
    # Corpus gewonnen, da sie jeweils die Gesamtzahl von Kategorien, Dokumenten
    # und Woertern ausgegeben haben.
    #
    # Geben Sie als naechstes die Anzahl von Dokumenten und Woertern je
    # Kategorie aus.
    # http://www.nltk.org/howto/corpus.html#categorized-corpora
    # Hilfreiche Funktionen: fileids, words 
    #
    # Visualisieren Sie die Verteilungen mit Hilfe von horizontalen bar plots.
    # http://matplotlib.org/examples/lines_bars_and_markers/barh_demo.html
    #
    # Optional: Plotten Sie die Verteilungen mit vertikalen bar plots.
    # Vermeiden Sie, dass sich die an der x-Achse aufgetragenen labels ueberlappen
    # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xticklabels
    # Stellen Sie nun die Verteilungen ueber Dokumente und Woerter in einem 
    # gemeinsamen Plot dar. Verwenden Sie unterschiedliche Farben.
    # http://matplotlib.org/examples/api/barchart_demo.html
     
    cat_to_count = {cat : (len(brown.fileids(categories = cat)), 
                           len(brown.words(categories = cat))) 
                    for cat in brown_categories}
    print('Kategorien und ihre Werte (Documente, Woerter)\n {}'.format(cat_to_count))
      
    y_pos = np.arange(len(brown_categories))
    cats = np.array(cat_to_count.keys())
    files = np.array(cat_to_count.values())[:,0]
    words = np.array(cat_to_count.values())[:,1]
      
    # plotting bars for documents
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats) 
    ax.invert_yaxis()
    ax.barh(y_pos, files,
            align='center', color='green')
    plt.show()
      
    # plotting bars for words
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats) 
    ax.invert_yaxis()
    ax.barh(y_pos, words,
            align='center', color='green')
    plt.show()

    
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
   
    
    # Um Texte / Dokumente semantisch zu analysieren, betrachtet man Verteilungen
    # ueber Wortvorkommen. Ziel dieser semantischen Analyse soll es letztlich sein
    # unbekannte Dokumente automatisch einer bekannten Kategorie / Klasse zuzuordnen.
    #
    
    # Bestimmen Sie die 20 haeufigsten Woerter des Brown Corpus (insgesamt), sowie
    # die 20 haeufigsten Woerter je Kategorie. 
    # http://docs.python.org/2/library/collections.html#collections.defaultdict
    # http://docs.python.org/2/library/functions.html#sorted
    # Hinweis: Die Dokumentation zu defaultdict enthaelt ein sehr hilfreiches Beispiel. 
    #
    # Implementieren Sie die (statische) Funktion BagOfWords.most_freq_words im Modul
    # features.
    
    w_most_20 = BagOfWords.most_freq_words(brown_words, n_words=20)
    cat_to_w_most_20 = {cat : BagOfWords.most_freq_words(brown.words(categories=cat), n_words=20)
                        for cat in brown_categories}
    
    print(w_most_20)
    print(cat_to_w_most_20)
    
    
    #
    # Diese Woerter sind nicht besonders charakteristisch fuer die Unterscheidung 
    # verschiedener Kategorien. Daher entfernt man solche wenig aussagekraeftigen
    # Woerter vor einer semantischen Analyse. Man bezeichnet diese Woerter als
    # stopwords.
    # Eine Liste mit stopwords wird durch NLTK bereitgestellt (siehe oben sowie 
    # im 'corpus' Modul). 
    # Filtern Sie nun alle stopwords bevor Sie die 20 haeufigsten Woerter im Brown
    # Corpus (insgesamt und je Kategorie) erneut bestimmen. Achten Sie dabei auch
    # Gross- und Kleinschreibung und filtern Sie ach Satzzeichen (string.punctuation). 
    # http://www.nltk.org/howto/corpus.html#word-lists-and-lexicons
    # http://docs.python.org/2/library/string.html
    #
    # Geben Sie zunaechst stopwords und Satzzeichen auf der Kommandozeile aus.
#     print(stopwords.words())
#     print(string.punctuation)
    
    # Mit der Liste von stopwords koennen Sie noch keine grammatikalischen Varianten
    # von Woertern erfassen, die ebenfalls nicht entscheidend fuer die semantische
    # Analyse von Texten sind (zum Beispiel: walking, walked).
    #
    # Verwenden Sie daher den PorterStemmer um Woerter auf ihre Wortstaemme abzubilden. 
    # Geben Sie die 20 haeufigsten Woerter nach jedem Filter Schrift aus: 
    #  1. stopwords und Satzzeichen
#     filtered_words = (w for w in brown_words
#                       if w not in stopwords.words() and w not in string.punctuation)
#     #print(BagOfWords.most_freq_words(filtered_words, n_words=20))
#     #  2. Abbildung auf Wortstaemme (stemming) 
#     stemmer = PorterStemmer()
#     stemmed_words = (stemmer.stem(w) for w in brown_words)
#     print(BagOfWords.most_freq_words(stemmed_words, n_words=20))
#     
    normalizer = WordListNormalizer()
    for c in brown_categories:
        w_cat = normalizer.normalize_words(brown.words(categories = c))
        print(c)
        print(BagOfWords.most_freq_words(w_cat[0],20))
        print(BagOfWords.most_freq_words(w_cat[1],20))
    # Erlaeutern Sie Ihre Beobachtungen.
    # http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.porter
    #
    # Implementieren Sie die Funktion WordListNormalizer.normalize_words im
    # features Modul.
    
    
    
    return


if __name__ == '__main__':
    print datetime.datetime.now()
    print '---\n'
    
    aufgabe1()
    
    print '\n---'
    print datetime.datetime.now()
