'''
Created on Nov 24, 2011

@author: arnaud
'''
import nltk
import numpy as np
import cPickle as pickle

#text1 = open('../languages/training_set1_10000.set').readlines()
text1 = pickle.load(open('../languages/generated'))

sentences2 = pickle.load(open('../languages/tong6words.enum'))
text2 = []
for s in sentences2:
    text2.extend(s)

print text1[:100]
print text2[:100]
   
D_tab = []
for l in range(1,6):
    
    ingrams  = list(nltk.util.ingrams(text1,l))
    ngrams1 = set(ingrams)
    freq1 = nltk.FreqDist(ingrams)
                          
    ingrams  = list(nltk.util.ingrams(text2,l))
    ngrams2 = set(ingrams)
    freq2 = nltk.FreqDist(ingrams)
    
    common_ngrams = ngrams1.intersection(ngrams2)
    unique_ngrams = ngrams1.difference(ngrams2)
    
    D = 0
    for ngram in common_ngrams:
        D += freq1.freq(ngram)*np.log(freq1.freq(ngram)/freq2.freq(ngram))
#    for ngram in unique_ngrams:
#        D += freq1.freq(ngram)*np.log(freq1.freq(ngram)/(3**(-l)))
    
    D_tab.append(D)
    
    print l, D, float(len(unique_ngrams))/len(ngrams1)*100, '%'

import pylab

pylab.plot(range(1, len(D_tab)+1), D_tab)
pylab.title("Kullback-Leibler divergence")
pylab.show()


