'''
Created on Nov 17, 2011

@author: arnaud
'''
import nltk
import random
import cPickle as pickle

"""Load the training examples"""
f = open('../languages/tong6words.enum', 'r')
sentences = pickle.load(f)

words = []
for sentence in sentences:
    words.extend(sentence)

l = 3
lm = nltk.model.NgramModel(l, words) 
print lm

print ""
for k in range(10):
    s = ''
    while len(s) < l:
        s = sentences[random.randint(0,len(sentences)-1)]
    
    print ' '.join(s)
    print "p(\'"+str(s[l-1])+"\'|\'"+str(' '.join(s[:l-1]))+"\') = ", lm.prob(s[l-1],s[:l-1])
