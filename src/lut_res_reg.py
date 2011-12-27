'''
Created on Dec 20, 2011

@author: arnaud
'''
import numpy as np
import Oger as og
import mdp
from my_utils import draw, setVocLut, sparseReservoirMatrix, similarity
import cPickle as pickle

vocabulary, lut = setVocLut()
f = open('../languages/tong6words.enum')
sentences = pickle.load(f)
for k in range(len(sentences)):
    sentences[k].insert(0,'.')

features = np.random.rand(len(vocabulary), 2)

in_d = features.shape[1]
out_d = 100
sw_reservoir = sparseReservoirMatrix((out_d,out_d), 0.27)

###
#f = open('../models/matrix_trop_cool', 'r')
#sw_reservoir = pickle.load(f)
#sw_in = pickle.load(f)
###

reservoir = og.nodes.ReservoirNode(input_dim=in_d
                                    ,output_dim=out_d
                                    ,input_scaling=1
                                    ,reset_states=False
#                                    ,w_in=sw_in
                                    ,w=sw_reservoir
                                    ,spectral_radius=0.97)

#f = open('../models/matrix_trop_cool', 'w')
#pickle.dump(sw_reservoir, f)
#pickle.dump(reservoir.w_in, f)
#f.close()
#print '... saved'

readout = np.random.rand(reservoir.output_dim, len(vocabulary))/reservoir.output_dim

print "... building dataset"
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
    reservoir.execute(np.dot(b, features))
initial_state = reservoir.states[-1]

##########
print "... similarity",
f = open('../models/similarity6', 'r')
distrib = pickle.load(f)  

esn_distrib = []
for s in sentences:
    reservoir.states = np.c_[initial_state].T
    for i in range(len(s)-1):
        x = reservoir.execute(np.dot(np.c_[lut[s[i]]].T, features))[0]
        x = np.dot(x, readout)
        x[np.nonzero(x<0)] = 0
        if sum(x) == 0:
            x[0] = 0.0001
        x = x/sum(x)
        esn_distrib.append(list(x))
print np.mean(similarity(distrib, esn_distrib))

import pylab
pylab.ion()
fig1 = pylab.figure(1)
pylab.draw()
pylab.title('Before')
ax1 = fig1.add_subplot(111, autoscale_on=True)
ax1.plot(features[:,0],features[:,1],'x')

pca = mdp.nodes.PCANode(output_dim=2)
pca.train(features)
pca.stop_training()
reduced_features = pca.execute(features)

for w in vocabulary:
    c = features[vocabulary.index(w)]
    ax1.annotate(w, (c[0],c[1]), xytext=(c[0],c[1]), xycoords='data', textcoords='offset points', arrowprops=None)

pylab.draw()
#########

print "... gradient descent"
momentum = 0
lr_decay = 0.99
lr = 1.
for t in range(100):
    lr *= lr_decay
    for k in range(len(source)):
        reservoir.states = np.c_[initial_state].T
        res = reservoir.execute(np.dot(source[k], features))
        output = np.dot(res, readout)
        for i in range(output.shape[0]):
            e = output[i] - target[k][i]
            p1 = np.dot(e, readout.T)
            p2 = 1-res[i]**2
            gradf = np.array([np.dot(p1*p2, reservoir.w_in)])*np.array([source[k][i]]).T
            features += -.005*lr*gradf
            gradr = np.array([e])*np.array([res[i]]).T
            readout += -.001*lr*(gradr + 0.9*momentum)
            momentum = gradr

########
            
    esn_distrib = []
    for s in sentences:
        reservoir.states = np.c_[initial_state].T
        for i in range(len(s)-1):
            x = reservoir.execute(np.dot(np.c_[lut[s[i]]].T, features))[0]
            x = np.dot(x, readout)
            x[np.nonzero(x<0)] = 0
            if sum(x) == 0:
                x[0] = 0.0001
            x = x/sum(x)
            esn_distrib.append(x)
    
    print "Epoch", t,
    print "    Similarity", np.mean(similarity(distrib, esn_distrib))

    fig = pylab.figure(1)
    pylab.title('After'+str(out_d))
    ax = fig1.add_subplot(111, autoscale_on=True)
    ax.plot(features[:-2,0],features[:-2,1],'x')
    
    pca = mdp.nodes.PCANode(output_dim=2)
    pca.train(features)
    pca.stop_training()
    reduced_features = pca.execute(features)
    
    for w in vocabulary[:-2]:
        c = features[vocabulary.index(w)]
        ax.annotate(w, (c[0],c[1]), xytext=(c[0],c[1]), xycoords='data', textcoords='offset points', arrowprops=None)
    pylab.draw()
    pylab.clf()

    import Oger
    
    read = Oger.nodes.RidgeRegressionNode()
       
    for k in range(len(source)):
        reservoir.states = np.c_[initial_state].T
        res = reservoir.execute(np.dot(source[k], features))
        read.train(res, target[k])
    read.stop_training()
    
    esn_distrib = []
    for s in sentences:
        reservoir.states = np.c_[initial_state].T
        for i in range(len(s)-1):
            x = reservoir.execute(np.dot(np.c_[lut[s[i]]].T, features))
            x = read.execute(x)[0]
            x[np.nonzero(x<0)] = 0
            if sum(x) == 0:
                x[0] = 0.0001
            x = x/sum(x)
            esn_distrib.append(x)
    
    print "Epoch", t,
    print "    Similarity", np.mean(similarity(distrib, esn_distrib))

        
        