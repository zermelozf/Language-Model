'''
Created on Dec 13, 2011

@author: arnaud
'''
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

train_set = []
for k in range(len(source)):
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
f = open('../deep_training/textmodel6', 'w')
g = open('../deep_training/textset6', 'w')
pickle.dump(reservoir.w_in, f, -1)
pickle.dump(reservoir.w_bias, f, -1)
pickle.dump(reservoir.w, f, -1)
pickle.dump(((train_set, train_label),(valid_set,valid_label), (test_set, test_label)), g, -1)
f.close()
g.close()

print "Done in", time.time()-tstart, "seconds."

print len(train_label), "training words."
print len(valid_label), "validation words."
print len(test_label), "test words."
