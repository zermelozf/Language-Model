'''
Created on Dec 14, 2011

@author: arnaud
'''
from my_utils import setVocLut, draw

vocabulary, lut = setVocLut()

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

import cPickle as pickle
import numpy as np

f = open('../deep_training/reservoir10_sentenceset6')

((train_set, train_label),(valid_set, valid_label),(test_set, test_label)) = pickle.load(f)

alldata = ClassificationDataSet(10, nb_classes=24)

for i in range(len(train_label)):
    alldata.addSample(train_set[i], [train_label[i]])

tstdata, trndata = alldata.splitWithProportion( 0.0 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
#print trndata['input'][0], 
print trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 500, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.9,  lrdecay=0.99999, verbose=True)

f = open('../models/similarity6', 'r')
distrib = pickle.load(f)

def similarity(d1, d2):
    normx = np.sqrt(np.sum(np.array(d1)**2, axis=1))
    normy = np.sqrt(np.sum(np.array(d2)**2, axis=1))
    prod = np.sum(np.array(d1)*np.array(d2),axis=1)
    return prod/(normx*normy)

approx = []
for k in range(len(train_label)):
    dist = fnn.activate(trndata['input'][k])
    approx.append(dist)
print "Initial Similarity:", np.mean(similarity(distrib, approx))

print "... training"
for t in range(200):
    trainer.trainEpochs(1)
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult
    
    approx = []
    for k in range(len(train_label)):
        dist = fnn.activate(trndata['input'][k])
        approx.append(dist)
    
#    print "Similarity:", similarity(distrib, approx)[:50]
    print "Mean Similarity:", np.mean(similarity(distrib, approx))
        
    
