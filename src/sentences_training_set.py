'''
Created on Dec 6, 2011

@author: arnaud
'''

import numpy as np
import Oger as og
from my_utils import setVocLut, sparseReservoirMatrix
import cPickle as pickle
import time
tstart = time.time()

vocabulary, lut = setVocLut()

f = open('../languages/tong6words.enum')
sentences = pickle.load(f)
print "... loaded", len(sentences), "sentences."
for k in range(len(sentences)):
    sentences[k].insert(0,'.')

in_d = len(vocabulary)
out_d = 500
w_reservoir = sparseReservoirMatrix((out_d,out_d), 0.27)

reservoir = og.nodes.ReservoirNode(input_dim=in_d
                                    ,output_dim=out_d
                                    ,input_scaling=1
                                    ,reset_states=False
                                    #,w_in=w_input
                                    ,w=w_reservoir
                                    ,spectral_radius=0.97)

""" Build dataset """
print "... building dataset"
source = []
train_label = []
for s in sentences:
    words = []
    nwordidx = []
    for k in range(len(s)-1):
        words.append(lut[s[k]])
        nwordidx.append(vocabulary.index(s[k+1]))
    source.append(np.r_[words])
    train_label.append(nwordidx)

for b in source[:10]:
    reservoir.execute(b)
initial_state = reservoir.states[-1]

train_set = []
for k in range(len(source)):
    reservoir.states = np.c_[initial_state].T
    train_set.append(reservoir.execute(source[k]))


assert len(train_label) == len(train_set)

#valid_set = []
#valid_label = []
#test_set = []
#test_label = []
#for k in range(250):
#    idx = np.random.randint(0,len(train_set))
#    test_set.append(train_set.pop(idx))
#    test_label.append(train_label.pop(idx))
#    
#    idx = np.random.randint(0,len(train_set))
#    valid_set.append(train_set.pop(idx))
#    valid_label.append(train_label.pop(idx))

train_set = np.vstack(train_set)
tampon = []
for l in train_label:
    tampon.extend(l)
train_label = tampon

valid_set = train_set
valid_label = train_label
test_set = train_set
test_label = train_label
  
#valid_set = np.vstack(valid_set)
#tampon = []
#for l in valid_label:
#    tampon.extend(l)
#valid_label = tampon
#
#test_set = np.vstack(test_set)
#tampon = []
#for l in test_label:
#    tampon.extend(l)
#test_label = tampon
#
#assert len(train_label) == train_set.shape[0]
#assert len(valid_label) == valid_set.shape[0]
#assert len(test_label) == test_set.shape[0]


""" Save """
print "... saving"
f = open('../deep_training/sentencemodel6', 'w')
g = open('../deep_training/sentenceset6', 'w')
pickle.dump(reservoir.w_in, f, -1)
pickle.dump(reservoir.w_bias, f, -1)
pickle.dump(reservoir.w, f, -1)
pickle.dump(initial_state, f, -1)
pickle.dump(((train_set, train_label),(valid_set,valid_label), (test_set, test_label)), g, -1)
f.close()
g.close()

print "Done in", time.time()-tstart, "seconds."

print len(train_label), "training words."
print len(valid_label), "validation words."
print len(test_label), "test words."







#""" Testing """
#print "... testing"
#
#print "... building labels"
#labels = []
#for l in train_label:
#    labels.append(np.eye(24)[l])
#
#labels = np.vstack(labels)
#
#assert labels.shape[0] == train_set.shape[0]
#
#print "... training"
#readout = og.nodes.RidgeRegressionNode()
#
#readout.train(train_set, labels)
#readout.stop_training()
#
#print "Matrix shape:", readout.beta.shape
#
#print "... constructing reservoir"
#f = open('../deep_training/model6', 'r')
#sw_in = pickle.load(f)
#sw_bias = pickle.load(f)
#sw = pickle.load(f)
#initial_state = pickle.load(f)
#f.close()
#
#reservoir = og.nodes.ReservoirNode(input_dim=in_d
#                                    ,output_dim=w_reservoir.shape[0]
#                                    ,input_scaling=1
#                                    ,reset_states=False
#                                    ,w=sw
#                                    ,w_in=sw_in
#                                    ,w_bias=sw_bias
#                                    ,spectral_radius=0.97)
#
#""" Produce a new text """
#print "... draw text"
#
#import pylab
#
#input = '.'
#text = [input]
#for k in range(15):
#    if input == '.':
#        reservoir.states = np.c_[initial_state].T
#    input = reservoir.execute(np.c_[lut[input]].T)
#    input = readout.execute(input)
#    pylab.figure(k)
#    pylab.title(' '.join(text[-6:]))
#    pylab.xticks(range(len(vocabulary)),vocabulary, rotation=70, size='small')
#    pylab.plot(input[0])
#    input = vocabulary[draw(input[0],1)]
#    text.append(input)
#pylab.show()