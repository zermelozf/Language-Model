'''
Created on Nov 25, 2011

@author: arnaud
'''
import random
import numpy as np
import mdp
import scipy.sparse as ssp
import Oger as og

vocabulary = ['John','Mary','boy','girl','cat','dog','boys','girls','cats','dogs',
              'feeds','walks','lives','hits','sees','hears',
              'feed','walk','live','hit','see','hear',
              'who','.']

lut = dict([(vocabulary[i],  0.5*np.eye(len(vocabulary))[i]) for i in range(len(vocabulary))])

def similarity(d1, d2):
    normx = np.sqrt(np.sum(np.array(d1)**2, axis=1))
    normy = np.sqrt(np.sum(np.array(d2)**2, axis=1))
    prod = np.sum(np.array(d1)*np.array(d2),axis=1)
    return prod/(normx*normy)

def setVocLut():
    return vocabulary, lut

def tokenize(text):
    words = []
    for sentence in text:
        words.extend(sentence[:-2].split(' '))
    return words

def draw(x,k):
    x[x<0] = 0.
    x = x**k
    x = x/sum(x)
    f = []
    for i in range(0,len(x)):
        f.append(sum(x[0:i+1]))
    z = random.random()
    f = np.array(f)
    return np.nonzero(f>=z)[0].min()

def addNoise(x, delta):
    x = x + np.random.uniform(-delta, delta, x.shape) 
    return x

def delayInputMatrix(in_dim, nb_delays):
    Id = np.eye(in_dim)
    zeros = np.zeros((in_dim, in_dim*(nb_delays-1)))
    w = np.hstack((Id,zeros))
    return w.T
    
def delayReservoirMatrix(in_dim, nb_delays):
    z1 = np.zeros((in_dim*(nb_delays-1), in_dim))
    Id = np.eye(in_dim*(nb_delays-1))
    h = np.hstack((z1,Id))
    l = np.zeros((in_dim, in_dim*nb_delays))
    return np.vstack((h,l))

import copy
def sparseReservoirMatrix(shape, d):
    w1 = np.random.rand(shape[0], shape[1])
    w2 = copy.copy(w1)
    w1[w1<(1-float(d)/2)] = 0.
    w2[w2>float(d)/2] = 0.
    w = w1+w2
    mask = np.array(w)
    mask[mask>0] = 1
    w = mask - 2*w
    w[w>0] = 1
    w[w<0] = -1
    w = 0.97*w/og.utils.get_spectral_radius(w)
    return w