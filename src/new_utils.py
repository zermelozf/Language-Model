'''
Created on Dec 21, 2011

@author: arnaud
'''
import numpy, cPickle, gzip, os

import theano
import theano.tensor as T

def load_data(dataset, nb_label):

    print '... loading data'
    # Load the dataset 
    f = open(dataset,'r')
    train_set = cPickle.load(f)
    f.close()


    def shared_dataset(data_xy, nb_label):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        z = []
        n_out = nb_label
        for y in data_y:
            t = numpy.zeros((1,n_out))
            t[0][y] = 1
            z.append(t)
        data_z = numpy.vstack(z)
        shared_z = theano.shared(numpy.asarray(data_z, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are 
        # floats it doesn't make sense) therefore instead of returning 
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), shared_z
    
    train_set_x, train_set_y, train_set_z = shared_dataset(train_set, nb_label)

    rval = (train_set_x, train_set_y, train_set_z)
    return rval