'''
Created on Dec 8, 2011

@author: arnaud
'''
import nltk
import numpy as np
import cPickle as pickle
from my_utils import setVocLut

def sentences_ngrams(sentences):
    ingrams = []
    for s in sentences:
        for l in range(len(s)+1):
            ingrams.extend(list(nltk.util.ingrams(s,l)))
        
    ngrams = set(ingrams)
    freq = nltk.FreqDist(ingrams)
    return ngrams, freq

f = open('../languages/tong6words.enum')
sentences = pickle.load(f)
for k in range(len(sentences)):
    sentences[k].insert(0,'.')
print "... Loaded", len(sentences), "sentences"

print "... building ngrams"
ngrams, freq = sentences_ngrams(sentences)


### Compute real probability distribution
vocabulary, lut = setVocLut()

print "... computing real distrib"
pos = []
distrib = []
for s in sentences:
    for i in range(len(s)-1):
        t1 = tuple(s[0:i+1])
        z = freq.freq(t1)
        if i == 0:
            z = z/4
        x = []
        for v in vocabulary:
            t2 = list(t1)
            t2.append(v)
            x.append(freq.freq(tuple(t2))/z)
        pos.append(t1)
        distrib.append(x)

### Compute ESN distribution
import Oger as og
import mdp
from my_utils import sparseReservoirMatrix

in_d = len(vocabulary)
out_d = 100
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

flow = mdp.Flow([reservoir, readout], verbose=1)
flownode = mdp.hinet.FlowNode(flow)

print "... training flownode"
source = []
target = []
for s in sentences:
    words = []
    nwords = []
    for k in range(len(s)-1):
        words.append(lut[s[k]])
        nwords.append(lut[s[k+1]])
    source.append(np.r_[words])
    target.append(np.r_[nwords])

for b in source[:10]:
    reservoir.execute(b)
initial_state = reservoir.states[-1]

for k in range(len(source)):
    reservoir.states = np.c_[initial_state].T
    flownode.train(source[k], target[k])
flownode.stop_training()

print "... computing esn distrib"
esn_distrib = []
for s in sentences:
    reservoir.states = np.c_[initial_state].T
    for i in range(len(s)-1):
        x = flownode.execute(np.c_[lut[s[i]]].T)[0]
        x[np.nonzero(x<0)] = 0
        x = x/sum(x)
        esn_distrib.append(x)

assert len(esn_distrib)== len(pos)
assert len(pos) == len(distrib)

def similarity(d1, d2):
    normx = np.sqrt(np.sum(np.array(d1)**2, axis=1))
    normy = np.sqrt(np.sum(np.array(d2)**2, axis=1))
    prod = np.sum(np.array(d1)*np.array(d2),axis=1)
    return prod/(normx*normy)
    
print "Similarity:", similarity(distrib, esn_distrib)[1000:1050]
print "Mean similarity:", np.mean(similarity(distrib, esn_distrib))

f = open('../models/similarity6', 'w')
pickle.dump(distrib, f, -1)

#""" Draw beautiful graphs """
#import pylab
#p = tuple(['.','boy','who','walks','sees','cat'])
#idx = pos.index(p)
#for k in range(idx-len(p)+1,idx+15):
#    pylab.figure(k)   
#    line = []    
#    line.append(pylab.plot(distrib[k]))
#    pylab.setp(line[0], linewidth=2, color='r', linestyle='--')
#    line.append(pylab.plot(esn_distrib[k]))
#    pylab.setp(line[1], linewidth=1, color='b')
#    pylab.title(str(pos[k]))
#    pylab.xticks(range(len(vocabulary)),vocabulary, rotation=70, size='small')
#    pylab.figlegend(line, ('real', 'esn'), 'upper right')
#pylab.show()
