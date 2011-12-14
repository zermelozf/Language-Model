'''
Created on Dec 7, 2011

@author: arnaud
'''
import numpy as np
import Oger as og
import mdp
import random
from my_utils import sparseReservoirMatrix, setVocLut
import cPickle as pickle

vocabulary, lut = setVocLut()


f = open('../languages/tong7words.enum')
sentences = pickle.load(f)
for k in range(len(sentences)):
    sentences[k].insert(0,'.')

in_d = len(vocabulary)
out_d = 200
w_reservoir = sparseReservoirMatrix((out_d,out_d), 0.27)

reservoir = og.nodes.ReservoirNode(input_dim=in_d
                                    ,output_dim=out_d
                                    ,input_scaling=1
                                    ,reset_states=False
                                    ,w=w_reservoir
                                    ,spectral_radius=0.97)

readout = og.nodes.RidgeRegressionNode(input_dim=reservoir.output_dim
                                       ,output_dim=len(vocabulary)
                                       ,ridge_param=0.)

print "... collecting"
source = []
label = []
p_label = []
for s in sentences:
    words = []
    for k in range(len(s)-1):
        words.append(lut[s[k]])
        label.append(vocabulary.index(s[k]))
        p_label.append(vocabulary.index(s[k-1]))
    source.append(np.r_[words])

for b in source[:10]:
    reservoir.execute(b)
initial_state = np.c_[reservoir.states[-1]].T

states = []
for k in range(len(source)):
    reservoir.states = initial_state
    reservoir.execute(source[k])
    states.append(reservoir.states)

states = np.vstack(states)

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
