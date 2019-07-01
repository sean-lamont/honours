#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

from enum import Enum, unique
import abc, six, os, sys, uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity( tf.logging.ERROR )

EPS = 1e-8   # to avoid log(0)

@unique
class Alg( Enum ):
    '''
    dimension reduction (DR) algorithms
    '''
    PCA            = 1
    TSNE           = 2

    CLRPCA         = 3
    CLRPCANN       = 4

    CODAPCA        = 5
    SCODAPCA       = 6
    CLRAE          = 7
    CODAAE         = 8
    NONPARACODAPCA = 9    # BFGS instead of SGD

def clip_grad_norms( gradients_to_variables, max_norm=10 ):
    '''
    remove NANs in the gradients and
    clip gradient norm to 10
    '''
    for grad, var in gradients_to_variables:
        if grad is not None:
            grad = tf.where( tf.is_nan(grad), tf.zeros(grad.shape), grad )
            if isinstance( grad, tf.IndexedSlices ):
                tmp = tf.clip_by_norm( grad.values, max_norm )
                grad = tf.IndexedSlices( tmp, grad.indices, grad.dense_shape )
            else:
                grad = tf.clip_by_norm( grad, max_norm )
        yield( (grad, var) )

def clr_transform( X ):
    '''
    centered log-ratio transformation of each row
    '''
    _X  = np.log( X + EPS )
    _X -= _X.mean( 1, keepdims=True )

    return _X

@six.add_metaclass( abc.ABCMeta )
class BaseDR():
    '''
    dimensionality reduction algorithm
    '''

    def __init__( self, n_components, alg, bias ):
        self.dtype        = tf.float32

        self.n_components = n_components
        self.alg          = alg
        self.bias         = bias

    @abc.abstractmethod
    def fit( self, X ):
        '''
        return the fitting error
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def transform( self, X ):
        '''
        X into the latent space
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def fit_transform( self, X ):
        '''
        fit and transform X
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform( self, A ):
        '''
        latent space to X
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def project( self, X ):
        '''
        compute the reconstruction of X, that is
        inverse_transform( transform( X ) )
        '''
        raise NotImplementedError()

class PCA( BaseDR ):
    '''
    Vanilla PCA
    '''

    def __init__( self, n_components, bias=True ):
        super( self.__class__, self ).__init__( n_components, Alg.PCA, bias )

    def fit( self, X ):
        dim = X.shape[1]
        assert( self.n_components < dim )

        if self.bias:
            self.b = X.mean( 0 )
        else:
            self.b = np.zeros( dim )

        # PCA is implemented by numpy's SVD
        _A, _w, V = np.linalg.svd( X-self.b, full_matrices=False )
        for _a, _b in zip( _w, _w[1:] ): assert( _a >= _b )

        self.V = V[:self.n_components]
        return np.linalg.norm( _w[self.n_components:] )

    def transform( self, X ):
        return ( X - self.b ).dot( self.V.T )

    def fit_transform( self, X ):
        self.fit( X )
        return self.transform( X )

    def inverse_transform( self, A ):
        return np.dot( A, self.V ) + self.b

    def project( self, X ):
        return self.inverse_transform( self.transform( X ) )

class TSNE( BaseDR ):
    '''
    tSNE
    '''

    def __init__( self, n_components=2, perplexity=30.0 ):
        super( self.__class__, self ).__init__( n_components, Alg.TSNE, True )

        from sklearn.manifold import TSNE
        self.tsne = TSNE( n_components=n_components, perplexity=perplexity )

    def fit( self, X ):
        self.tsne.fit( X )
        return self.tsne

    def transform( self, X ):
        return None

    def fit_transform( self, X ):
        return self.tsne.fit_transform( X )

    def inverse_transform( self, A ):
        return None

    def project( self, X ):
        return None

class CLRPCA( BaseDR ):
    '''
    CLR transformation -> PCA

    PCA is implemented based on numpy's SVD
    '''
    def __init__( self, n_components, bias=True ):
        super( self.__class__, self ).__init__( n_components, Alg.CLRPCA, bias )

    def fit( self, X, repeat=1, verbose=True ):
        '''
        fit clr(X) with PCA
        '''
        dim = X.shape[1]
        assert( self.n_components < dim )

        clrX = clr_transform( X )

        if self.bias:
            self.b = clrX.mean( 0 )
            clrX = clrX - self.b
        else:
            self.b = np.zeros( dim )

        # call SVD multiple times and select the best
        pairs = [ self.__fit( clrX, verbose ) for _ in range( repeat ) ]
        pairs.sort( key=lambda pair: pair[0] )

        self.V = pairs[0][1]
        return pairs[0][0]

    def __fit( self, clrX, verbose ):
        _A, _w, V = np.linalg.svd( clrX, full_matrices=False )
        for _a, _b in zip( _w, _w[1:] ): assert( _a >= _b )

        return np.linalg.norm( _w[self.n_components:] ), V[:self.n_components]

    def transform( self, X ):
        clrX = clr_transform( X )
        return ( clrX - self.b ).dot( self.V.T )

    def fit_transform( self, X ):
        self.fit( X )
        return self.transform( X )

    def inverse_transform( self, A ):
        '''
        Notice:
        inverse_transfrom(A) is still in the CLR space
        one has to compute exp(inverse_transform)
        to transfer back to the positive measure space
        '''
        return np.dot( A, self.V ) + self.b

    def project( self, X ):
        return self.inverse_transform( self.transform( X ) )

class CodaPCA( BaseDR ):
    '''
    Parametric CoDA PCA by stochastic gradient descent
    '''

    def __init__( self,
                  n_components,
                  lrate,
                  nn_shape,
                  decode_shape=[100,],
                  batchsize=32,
                  alg=Alg.CODAPCA,
                  noise_level=0.1,
                  rand_seed=2019,
                  bias=True ):
        '''
        n_components -- number of PCs = dim(y)
        lrate        -- learning rate
        nn_shape     -- NN shape of the projection x->y
        decode_shape -- NN shape of the projection y->x (for AEs)
        batchsize    -- mini batch size for SGD
        alg          -- algorithm to be used
        noise_level  -- noise level for denoising autoencoder
        bias         -- whether to use a bias term
        rand_seed    -- random seed
        '''
        super( self.__class__, self ).__init__( n_components, alg, bias )
        assert( alg in [ Alg.CLRPCANN, Alg.CODAPCA, Alg.SCODAPCA, Alg.CLRAE, Alg.CODAAE ] )

        self.loss  = np.inf

        self.lrate        = lrate
        self.nn_shape     = nn_shape
        self.decode_shape = decode_shape
        self.noise_level  = noise_level
        self.batchsize    = batchsize
        self.rand_seed    = rand_seed

    def batches( self, allidx, batchsize ):
        '''
        generate mini-batches (for SGD)
        '''
        N = allidx.size
        if N < 2 * batchsize:
            yield allidx

        else:
            # have at least 2 mini-batches per epoch
            for i in range( int( np.ceil(N/batchsize) ) ):
                if (i+1)*batchsize <= N:
                    yield allidx[i*batchsize:(i+1)*batchsize]

                else:
                    yield np.hstack( [ allidx[i*batchsize:],
                                       allidx[:((i+1)*batchsize)%N] ] )

    def fit( self, X, repeat=1, epochs=100, verbose=True ):
        '''
        X_train is 2D array of shape N x dim (one sample per row)
        '''
        self.__rebuild_graph( X.shape[1] )
        tf.set_random_seed( self.rand_seed )

        saver = tf.train.Saver( max_to_keep=repeat )

        with tf.Session() as sess:
            pairs = [ self.__fit( sess, saver, X, epochs, verbose )
                      for _ in range( repeat ) ]
        pairs.sort( key=lambda pair: pair[0] )

        self.save_path = pairs[0][1]
        return pairs[0][0]

    def transform( self, X, sess=None ):
        '''
        X transformed into low-dimensional space (the A matrix)
        '''
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore( sess, self.save_path )

            if self.alg in ( Alg.CODAPCA, Alg.SCODAPCA, Alg.CLRPCANN ):
                # for these methods, one need to make A's columns orthogonal

                _A, _V = sess.run( [self.A, self.V], feed_dict={ self.X:X, self.noise:0 } )

                # Y = ( A V + b )C
                # orghogonize the basis
                _V -= _V.mean( 1, keepdims=True )
                _Q, _R = np.linalg.qr( _V.T )
                _A = _A.dot( _R.T )
            else:
                _A = sess.run( self.A, feed_dict={ self.X:X, self.noise:0 } )

        return _A

    def fit_transform( self, X, repeat=1, epochs=100, verbose=True ):
        '''
        fit the model to X and transform X
        '''
        self.fit( X, repeat=repeat, epochs=epochs, verbose=verbose )

        return self.transform( X )

    def inverse_transform( self, A ):
        raise NotImplementedError( 'not implemented; use project() instead' )

    def project( self, X ):
        '''
        Note: returned Y is in the CLR space
        '''
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore( sess, self.save_path )

            _Y = self.Y.eval( feed_dict={ self.X:X, self.noise:0 } )

        return _Y

    def __rebuild_graph( self, dim_x ):
        '''
        build the computational graph
        '''
        tf.reset_default_graph()

        with tf.variable_scope( 'CodaPCA' ):
            self.X     = tf.placeholder( self.dtype, (None, dim_x), 'X' )
            self.noise = tf.placeholder( self.dtype, (),        'noise' )

            mask   = tf.cast( self.X > EPS, tf.float32 )
            degree = tf.reduce_sum( mask, axis=1, keep_dims=True )

            logX = tf.log( tf.clip_by_value( self.X, EPS, 1 ) )
            #clrX = logX - tf.reduce_mean( logX, axis=1, keepdims=True )
            logMeanX = tf.reduce_sum( logX*mask, axis=1, keep_dims=True ) / degree
            clrX = logX - logMeanX

            #checkX = self.X * tf.exp( -tf.reduce_mean( logX, axis=1, keepdims=True ) )
            checkX = mask * self.X * tf.exp( -logMeanX )

            # In CodaPCA,
            # the low-dimensonal coordinate matrix A is assumed to be a non-linear transformation of X,
            # so that we learn a parametric mapping X->A instead of letting A free

            if self.alg in [ Alg.CODAAE, Alg.CLRAE ]:
                corrupt_logX = logX + self.noise * tf.random_normal( tf.shape(self.X), dtype=self.dtype )
                corrupt_clrX = corrupt_logX - tf.reduce_mean( corrupt_logX, axis=1, keep_dims=True )
                corrupt_X = tf.exp( corrupt_logX - tf.reduce_logsumexp( corrupt_logX, axis=1, keep_dims=True ) )
                _A = tf.concat( [ corrupt_X, corrupt_clrX ], axis=1 )
            else:
                _A = tf.concat( [self.X, clrX], axis=1 )

            for odim in self.nn_shape:
                _A = tf.layers.dense( _A, odim, activation=tf.nn.elu )
            self.A = tf.layers.dense( _A, self.n_components )

            # the linear transformation of PCA
            self.V = tf.get_variable( 'V', dtype=self.dtype, shape=(self.n_components, dim_x) )

            if self.bias:
                self.b = tf.get_variable( 'b', dtype=self.dtype, shape=(1,dim_x) )
            else:
                self.b = tf.constant( 0, dtype=self.dtype, shape=(1,dim_x), name='b' )

            if self.alg == Alg.SCODAPCA:
                '''
                see section 4.4
                '''
                self.Y  = tf.matmul( self.A, self.V ) + self.b
                #self.Y -= tf.reduce_mean( self.Y, 1, keepdims=True )
                self.Y -= tf.reduce_sum( mask*self.Y, 1, keep_dims=True ) / degree

                self.loss  = tf.reduce_sum( checkX * tf.exp(-self.Y), axis=1 ) * tf.reduce_sum( tf.exp(self.Y), axis=1 ) / degree
                self.loss -= tf.reduce_sum( checkX * self.Y, axis=1 )
                self.loss  = tf.reduce_mean( self.loss )

            elif self.alg == Alg.CLRPCANN:
                '''
                CLR+PCA (implemented by tensorflow)
                '''
                self.Y = tf.matmul( self.A, self.V ) + self.b
                self.Y -= tf.reduce_sum( mask*self.Y, 1, keep_dims=True ) / degree
                self.loss = tf.reduce_mean( tf.reduce_sum( tf.pow( self.Y - clrX, 2 ), 1 ) )

            elif self.alg == Alg.CODAPCA:
                '''
                Bregman divergence wrt exp()
                '''
                self.Y = tf.matmul( self.A, self.V ) + self.b
                #self.Y -= tf.reduce_mean( self.Y, 1, keepdims=True )
                self.Y -= tf.reduce_sum( mask*self.Y, 1, keep_dims=True ) / degree

                self.loss = tf.reduce_sum( tf.exp(self.Y) - checkX * self.Y, axis=1 )
                self.loss = tf.reduce_mean( self.loss )

            elif self.alg == Alg.CLRAE:
                '''
                squared loss based on non-linear transformation of A
                '''
                _Y = self.A
                for odim in self.decode_shape:
                    _Y = tf.layers.dense( _Y, odim, activation=tf.nn.elu )
                self.Y = tf.layers.dense( _Y, dim_x )
                #self.Y -= tf.reduce_mean( self.Y, 1, keepdims=True )
                self.Y -= tf.reduce_sum( mask*self.Y, 1, keep_dims=True ) / degree

                self.loss = tf.reduce_mean( tf.reduce_sum( tf.pow( self.Y - clrX, 2 ), 1 ) )

            elif self.alg == Alg.CODAAE:
                '''
                Bregman loss based on non-linear transformation of A
                '''
                _Y = self.A
                for odim in self.decode_shape:
                    _Y = tf.layers.dense( _Y, odim, activation=tf.nn.elu )
                self.Y = tf.layers.dense( _Y, dim_x )
                #self.Y -= tf.reduce_mean( self.Y, 1, keepdims=True )
                self.Y -= tf.reduce_sum( mask*self.Y, 1, keep_dims=True ) / degree

                self.loss = tf.reduce_mean( tf.reduce_sum( tf.exp(self.Y) - checkX * self.Y, axis=1 ) )

            else:
                raise RuntimeError( 'unknown algorithm' )

        optimizer     = tf.train.AdamOptimizer( learning_rate=self.lrate )
        self.train_op = optimizer.minimize( self.loss )

        #grads_vars = optimizer.compute_gradients( self.loss, tf.trainable_variables() )
        #grads_vars = clip_grad_norms( grads_vars )
        #self.train_op = optimizer.apply_gradients( grads_vars )

    def __fit( self, sess, saver, X, epochs, verbose ):
        '''
        training epochs
        have to be called within a session
        '''
        sess.run( tf.global_variables_initializer() )

        allidx = np.arange( X.shape[0] )
        for epoch in range( epochs ):
            np.random.shuffle( allidx )

            for batchidx in self.batches( allidx, self.batchsize ):
                sess.run( self.train_op, feed_dict={ self.X: X[batchidx],
                                                     self.noise: self.noise_level } )

            if verbose and ( epoch % 100 == 0 ):
                _loss = self.loss.eval( feed_dict={ self.X:X, self.noise:0 } )
                print( '[epoch {0:5d}] L={1:8.4f}'.format( epoch, _loss ) )

        modelname = uuid.uuid4()
        if not os.path.isdir( ".models" ): os.mkdir( ".models" )
        savepath = saver.save( sess, ".models/{}.ckpt".format( modelname ) )

        return self.loss.eval( feed_dict={ self.X:X, self.noise:0 } ), savepath

class NonParametricCodaPCA( BaseDR ):
    '''
    CodaPCA without learning the parametric mapping

    X->A
    '''

    def __init__( self, n_components, bias=True ):
        super( self.__class__, self ).__init__( n_components, Alg.NONPARACODAPCA, bias )

    def fit( self, X ):
        N, dim = X.shape
        assert( self.n_components < dim )

        checkX = X * np.exp( np.mean( -np.log( X + EPS ), axis=1, keepdims=True ) )

        def loss_func( _x ):
            # the loss function to be minimized
            U = _x[:(self.n_components+1)*dim].reshape( self.n_components+1, dim )
            U -= U.mean( 1, keepdims=True )

            B = _x[(self.n_components+1)*dim:].reshape( N, self.n_components )
            B = np.hstack( [ B, np.ones( [N,1] ) ] )

            Y = np.dot( B, U )

            return ( np.exp( Y ) - checkX * Y ).sum()

        def grad_func( _x ):
            # gradient of the loss function
            U = _x[:(self.n_components+1)*dim].reshape( self.n_components+1, dim )
            U -= U.mean( 1, keepdims=True )

            B = _x[(self.n_components+1)*dim:].reshape( N, self.n_components )
            B = np.hstack( [ B, np.ones( [N,1] ) ] )
            Y = np.dot( B, U )

            diff  = np.exp( Y ) - checkX
            gradB = np.dot( diff, U.T )[:,:-1]

            gradU  = np.dot( B.T, diff )
            gradU -= gradU.mean( 1, keepdims=True )

            return np.hstack( [ gradU.flatten(), gradB.flatten() ] )

        x0 = 1e-4 * np.random.randn( (self.n_components+1)*dim + N*self.n_components )
        result = minimize( loss_func, x0, method='L-BFGS-B', jac=grad_func )
        assert result.success, 'optimization failed'

        result = result.x
        self.U = result[:(self.n_components+1)*dim].reshape( self.n_components+1, dim )
        self.B = result[(self.n_components+1)*dim:].reshape( N, self.n_components )
        assert( np.allclose( self.U.sum(1), 0 ) )

    def transform( self, X ):
        return self.fit_transform( X )

    def fit_transform( self, X ):
        self.fit( X )
        return self.B

    def inverse_transform( self, A ):
        '''impossible to do inverse transform without parametric mapping'''
        return None

    def project( self, X ):
        self.fit( X )
        B = np.hstack( [ self.B, np.ones( [ X.shape[0], 1 ] ) ] )
        return np.dot( B, self.U )
