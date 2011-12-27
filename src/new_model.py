import theano
import theano.tensor as T
from new_utils import *
from my_utils import similarity, sparseReservoirMatrix
import time


f = open('../models/similarity6', 'r')
distrib = cPickle.load(f)
f.close()

class Features(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        
        W_values = numpy.random.rand(n_in, n_out)
        self.W = theano.shared(value = W_values, name ='W')
        
        self.output = T.dot(self.input, self.W)
        
        self.params = [self.W]

class ReservoirLayer(object):
    def __init__(self, input, n_in, n_out, activation = T.tanh):
        self.input = input

        W_in_values = 1. * (numpy.random.randint(0, 2, (n_in, n_out)) * 2 - 1)
        self.W_in = theano.shared(value = W_in_values, name ='W_in')
        
        W_res_values = sparseReservoirMatrix((n_out, n_out), 0.27)
        self.W = theano.shared(value = W_res_values, name ='W')

        b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
        self.b = theano.shared(value= b_values, name ='b')
        
        self.output = numpy.zeros((n_out,), dtype= theano.config.floatX)
        self.output = theano.shared(value= b_values, name ='out')
        self.output = activation(T.dot(input, self.W_in) + self.b + T.dot(self.output, self.W))
        
        self.params = [self.W_in, self.W, self.b]

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in,n_out), dtype = theano.config.floatX), name='W')
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype = theano.config.floatX), name='b')
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)
        
        self.params = [self.W, self.b]

    def mean_square_error(self, y):
        return T.mean((self.p_y_given_x - y)**2)

class new_model(object):
    def __init__(self, input, n_in, n_feat, n_hidden, n_out):
        
        self.features = Features(input, n_in, n_feat)
        
        self.reservoirLayer = ReservoirLayer(input = self.features.output, 
                                 n_in = n_feat, n_out = n_hidden,
                                 activation = T.tanh)

        self.logRegressionLayer = LogisticRegression( 
                                    input = self.reservoirLayer.output,
                                    n_in  = n_hidden,
                                    n_out = n_out)
                    
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        self.mean_square_error = self.logRegressionLayer.mean_square_error

        self.params = self.features.params #+ self.logRegressionLayer.params 
        
def test_model( n_in
                ,n_feat
                ,n_hidden
                ,n_out
                ,learning_rate=0.01
                , n_epochs=1000
                , dataset = '../deep_training/new_model_train6'
                , batch_size = 20):

    train_set_x, train_set_y, train_set_z = load_data(dataset, n_out)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'
    index = T.lscalar() 
    x     = T.matrix('x')
    z     = T.matrix('z')

    classifier = new_model(input=x, n_in=24, n_feat=5, n_hidden = 50, n_out=n_out)

    cost = classifier.mean_square_error(z) 

    gparams = []
    for param in classifier.params:
        gparam  = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a dictionary
    updates = {}
    for param, gparam in zip(classifier.params, gparams):
        updates[param] = param - learning_rate*gparam

    train_model =theano.function( inputs = [index], outputs = cost, 
            updates = updates,
            givens={
                x:train_set_x[index*batch_size:(index+1)*batch_size],
                z:train_set_z[index*batch_size:(index+1)*batch_size]})
    
    execute_model = theano.function(inputs = [index], 
        outputs = classifier.p_y_given_x,
        givens={x:train_set_x[index*batch_size:(index+1)*batch_size]})

    print '... training'
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        shuffled_idx = range(n_train_batches)
        numpy.random.shuffle(shuffled_idx)
        
        for minibatch_index in shuffled_idx:
            minibatch_avg_cost = train_model(minibatch_index)

        test_distrib = [execute_model(i) for i in xrange(n_train_batches)]
        test_distrib = numpy.vstack(test_distrib)
        approx = [t for t in test_distrib]
        avg_sim = numpy.mean(similarity(distrib, approx))

        print "Similarity:", avg_sim

    end_time = time.clock()
    print "... done in", end_time - start_time
    
if __name__ == '__main__':
    test_model(n_in = 24
              , n_feat = 5
              , n_hidden = 50
              , n_out = 24
              , learning_rate=1.
              , n_epochs=1000
              , dataset = '../deep_training/new_model_train6'
              , batch_size = 20)
    
    

