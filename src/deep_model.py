'''
Created on Dec 22, 2011

@author: arnaud
'''
import numpy
from my_utils import sparseReservoirMatrix
import mdp

import Oger
Oger.gradient.BackpropNode

def identity(x):
    return x

class FeaturesNode(mdp.Node):
    def __init__(self, input_dim, output_dim, dtype=None):
        super(FeaturesNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)  
        self.W = numpy.random.rand(input_dim, output_dim)
        
        self.params = [self.W]
        
    def is_trainable(self):
        return False
       
    def _execute(self, input):
        return numpy.dot(input, self.W)

class ReservoirNode(mdp.Node):
    def __init__(self, input_dim, output_dim, activation = numpy.tanh, dtype=None):
        super(ReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.activation = activation

        self.W_in = 1. * (numpy.random.randint(0, 2, (input_dim, output_dim)) * 2 - 1)
        self.b = numpy.zeros((output_dim,)) 
        self.W = sparseReservoirMatrix((output_dim, output_dim), 0.27)
        
        self.state = numpy.zeros((output_dim,))
        
        self.params = [self.W_in, self.W, self.b]
    
    def is_trainable(self):
        return False
        
    def _execute(self, input):
        self.state = self.activation(numpy.dot(input, self.W_in) + self.b + numpy.dot(self.state, self.W))
        return self.state

class HiddenLayerNode(mdp.Node):
    def __init__(self, input_dim, output_dim, activation = numpy.tanh, dtype=None):
        super(HiddenLayerNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.activation = activation

        self.W = numpy.asarray( numpy.random.uniform(
                low  = - numpy.sqrt(6./(input_dim+output_dim)),
                high = numpy.sqrt(6./(input_dim+output_dim)),
                size = (input_dim, output_dim)))

        self.b = numpy.zeros((output_dim,))

        self.params = [self.W, self.b]
    
    def is_trainable(self):
        return False
        
    def _execute(self, input):
        return self.activation(numpy.dot(input, self.W) + self.b)

class PerceptronNode(HiddenLayerNode):
    def __init__(self, input_dim, output_dim, activation=identity, dtype=None):
        super(PerceptronNode, self).__init__(input_dim=input_dim, output_dim=output_dim, activation=identity, dtype=dtype)

class SoftMaxNode(mdp.Node):
    def __init__(self, input_dim=None, dtype=None):
        super(SoftMaxNode, self).__init__(input_dim=input_dim, output_dim=input_dim, dtype=dtype)
        
    def is_trainable(self):
        return False
    
    def _execute(self, input):
        #input = numpy.exp(input)
        return input/numpy.array([numpy.sum(input, axis=1)]).T
    
if __name__=='__main__':
    
    feat = FeaturesNode(input_dim=10, output_dim=5)
    res = ReservoirNode(input_dim=feat.output_dim, output_dim=50)
    hid = HiddenLayerNode(input_dim=res.output_dim, output_dim=10, activation=identity)
    per = PerceptronNode(input_dim=hid.output_dim, output_dim=5)
    cut = mdp.nodes.CutoffNode(lower_bound=0)
    sof = SoftMaxNode(input_dim=cut.output_dim)
    
    readout = mdp.hinet.FlowNode(per + cut + sof)
    flow = mdp.Flow(feat + res + hid + readout)
    
    data = numpy.random.random((5,10))
    result = flow.execute(data)
    print result
    print numpy.sum(result[0], axis=0)
    mdp.hinet.show_flow(flow)
        
        
        