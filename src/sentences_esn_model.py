'''
Created on Dec 7, 2011

@author: arnaud
'''
import numpy as np
import Oger as og
import mdp
from my_utils import draw, setVocLut, sparseReservoirMatrix
import cPickle as pickle

vocabulary, lut = setVocLut()

f = open('../languages/tong6words.enum')
sentences = pickle.load(f)
for k in range(len(sentences)):
    sentences[k].insert(0,'.')

in_d = len(vocabulary)
out_d = 500
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

""" Train """
print "... training", flownode
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
    
""" Produce a new text """
print "Draw text..."

input = '.'
text = [input]
for k in range(500):
    if input == '.':
        reservoir.states = np.c_[initial_state].T
    input = flownode.execute(np.c_[lut[input]].T)
    input = vocabulary[draw(input[0],1)]
    text.append(input)

#f = open('../languages/generated','w')
#pickle.dump(text,f)  

print ' '.join(text)

import pylab
max = 15
for k in range(max):
    if input == '.':
        reservoir.states = np.c_[initial_state].T
    input = flownode.execute(np.c_[lut[input]].T)
    input[np.nonzero(input<0)] = 0
    pylab.figure(k)
    pylab.title(' '.join(text[np.max(-6,-len(text)):]))
    pylab.xticks(range(len(vocabulary)),vocabulary, rotation=70, size='small')
    pylab.plot(input[0])
    input = vocabulary[draw(input[0],1)]
    text.append(input)
pylab.show()
    
    