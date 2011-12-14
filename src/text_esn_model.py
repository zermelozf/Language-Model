'''
Created on Nov 22, 2011

@author: arnaud
'''
import numpy as np
import Oger as og
import mdp
import cPickle as pickle
from my_utils import draw, sparseReservoirMatrix, setVocLut

vocabulary, lut = setVocLut()


f = open('../languages/tong6words.enum')
sentences = pickle.load(f)

words = []
for sentence in sentences:
    words.extend(sentence)


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

flow = mdp.Flow([reservoir, readout], verbose=1)
flownode = mdp.hinet.FlowNode(flow)

""" Train """
print "Training", flownode, "..."
source = []
target = []
label = []
for i in range(len(words)):
    source.append(lut[words[i]])
    target.append(lut[words[(i+1)%len(words)]])
    label.append(vocabulary.index(words[(i+1)%len(words)]))
#    print words[i], words[(i+1)%len(words)] 

source = np.r_[source]
target = np.r_[target]


flownode.train(source, target)
flownode.stop_training()


""" Produce a new text """
print "Draw text..."

input = '.'
text = [input]
for k in range(500):
    input = flownode.execute(np.c_[lut[input]].T)
    input = vocabulary[draw(input[0],1)]
    text.append(input)

print ' '.join(text)

#import pylab
#
#input = words[0]
#text = [input]
#max = 30
#for k in range(max):
#    input = flownode.execute(np.c_[lut[input]].T)
#    input = input + 0.5
#    input[np.nonzero(input<0)] = 0
#    pylab.figure(k)
#    pylab.title(' '.join(text[np.max(-6,-len(text)):]))
#    pylab.xticks(range(len(vocabulary)),vocabulary, rotation=70, size='small')
#    pylab.plot(input[0])
#    input = vocabulary[draw(input[0],1)]
#    text.append(input)
#
#pylab.show()

