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
out_d = 50
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
    target.extend(np.r_[nwords])

for b in source[:10]:
    reservoir.execute(b)
initial_state = reservoir.states[-1]

states = []
for k in range(len(source)):
    reservoir.states = np.c_[initial_state].T
    states.append(reservoir.execute(source[k]))

X = np.vstack(states)

import mlpy

K = mlpy.kernel_gaussian(X.T, X.T, sigma=1)

readout = mlpy.KernelRidge(lmb=0.01)
readout.learn(K, np.array(target)[:,0])

kernel_distrib = readout.pred(X)

