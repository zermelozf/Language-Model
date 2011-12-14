'''
Created on Dec 7, 2011

@author: arnaud
'''
import numpy as np
import Oger as og
import mdp
import random
from my_utils import sparseReservoirMatrix, setVocLut

vocabulary, lut = setVocLut()

f = open('../languages/tong6words.enum')
#f = open('../languages/tr/aining_set1_10000.set')
sentences = f.readlines()

words = []
for sentence in sentences:
    words.extend(sentence[:-1].split(' '))


in_d = len(vocabulary)
out_d = 200
w_reservoir = sparseReservoirMatrix((out_d,out_d), 0.27)

reservoir = og.nodes.LeakyReservoirNode(input_dim=in_d
                                    ,output_dim=out_d
                                    ,input_scaling=1.
                                    ,reset_states=False
                                    ,w=w_reservoir
                                    ,spectral_radius=0.97
                                    ,leak_rate=1.)

print "... collecting"
source = []
label = []
p_label = []
for i in range(len(words)):
    source.append(lut[words[i]])
    label.append(vocabulary.index(words[i]))
    p_label.append(vocabulary.index(words[(i-1)]))
#    print words[i], words[(i+1)%len(words)] 

source = np.r_[source]
states = reservoir.execute(source)

print "... computing PCA"
pca = mdp.nodes.PCANode(output_dim=2)
pca.train(states)
pca.stop_training()

pca_states = pca.execute(states)

print "... display"
import pylab

#Word Regions
pylab.figure(1)
pylab.title('1-gram Regions')
line = []
for k in range(len(vocabulary)):
    idx = np.nonzero(np.array(label) == k)[0]
    col = (random.random(), random.random(), random.random())
    line.append(pylab.plot(pca_states[idx].T[0], pca_states[idx].T[1], 'x', color=col))
        
pylab.figlegend(line, vocabulary, 'upper right')

#2-grams Regions
w_idx = 22
pylab.figure(2)
pylab.title(vocabulary[w_idx]+' 2-grams Regions')
p_label = np.array(p_label)
p_label = p_label[np.nonzero(np.array(label)==w_idx)[0]]
pca_states = pca_states[np.nonzero(np.array(label)==w_idx)[0]]
line = []
for k in range(len(vocabulary)):
    idx = np.nonzero(np.array(p_label) == k)[0]
    col = (random.random(), random.random(), random.random())
    line.append(pylab.plot(pca_states[idx].T[0], pca_states[idx].T[1], 'x', color=col))

pylab.figlegend(line, vocabulary, 'upper right')
    
pylab.show()







