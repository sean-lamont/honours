"""
standard measurements for CoDA data

measure( X, P )
where X, P are positive measures (one sample per row)

all measurements are invariant to constant scaling of X and P
so that we have
measure( lambda_1 X, lambda_2 P ) = measure( X, P )
for all lambda_1 > 0, lambda_2 > 0
"""

from __future__ import absolute_import, print_function, division

from CodaPCA import clr_transform, EPS
import numpy as np

def perspective_kl( X, P ):
    '''
    KL between pserspective measures
    '''
    clrX = clr_transform( X )
    checkX = np.exp( clrX )

    clrP = clr_transform( P )
    checkP = np.exp( clrP )

    return ( ( checkX * ( clrX - clrP ) - checkX + checkP ).sum(1) ).mean()

def sym_perspective_kl( X, P ):
    '''
    symmetric perspective kl
    '''
    return ( 0.5 * perspective_kl(X,P) + 0.5 * perspective_kl(P,X) )

def kl( X, P ):
    '''
    kl between two probability measures
    '''
    normX = X / ( X.sum( 1, keepdims=True ) + EPS )
    normP = P / ( P.sum( 1, keepdims=True ) + EPS )

    return ( ( normX * ( np.log( normX + EPS ) - np.log( normP + EPS ) ) ).sum(1) ).mean()

def sym_kl( X, P ):
    '''
    symmetric KL
    '''
    return ( 0.5 * kl( X, P ) + 0.5 * kl( P, X ) )

def js( X, P ):
    '''
    Jensen Shannon divergence
    '''
    normX = X / ( X.sum( 1, keepdims=True ) + EPS )
    normP = P / ( P.sum( 1, keepdims=True ) + EPS )
    mixXP = 0.5 * ( normX + normP )

    return ( 0.5 * kl( normX, mixXP ) + 0.5 * kl( normP, mixXP ) )

def l2( X, P ):
    '''
    l2 norm in the CLR space
    '''
    clrX = clr_transform( X )
    clrP = clr_transform( P )

    return np.mean( np.linalg.norm( clrX-clrP, axis=1 ) )

def perspective_l2( X, P ):
    '''
    l2 norm in the exp-CLR space
    '''
    expX = np.exp( clr_transform( X ) )
    expP = np.exp( clr_transform( P ) )

    return np.mean( np.linalg.norm( expX-expP, axis=1 ) )

def simplex_l2( X, P ):
    '''
    l2 norm in the simplex space
    '''

    normX = X / ( X.sum( 1, keepdims=True ) + EPS )
    normP = P / ( P.sum( 1, keepdims=True ) + EPS )

    return np.mean( np.linalg.norm( normX-normP, axis=1 ) )

def tv( X, P ):
    '''
    total variance distance
    '''

    normX = X / ( X.sum( 1, keepdims=True ) + EPS )
    normP = P / ( P.sum( 1, keepdims=True ) + EPS )

    return 0.5 * np.mean( np.abs( normX - normP ).sum(1) )

def riemannian( X, P ):
    '''
    riemannian distance
    '''
    ballX = np.sqrt( X / ( X.sum( 1, keepdims=True ) + EPS ) )
    ballP = np.sqrt( P / ( P.sum( 1, keepdims=True ) + EPS ) )

    inner_product = np.clip( ( ballX * ballP ).sum(1), -1, 1 )
    return 2 * np.arccos( inner_product ).mean()

def compute_scores( X_train, Y_train, X_test, Y_test ):
    '''
    compute the scores based on given data

    X is the original Coda data (on the simplex)
    Y is the reconstruction (in the CLR space)
    '''

    _names = [ 'SPKL (train)', 'JSD (train)', r'$L^2$-clr (train)', r'$L^2$ (train)', r'TV (train)', r'Riemannian (train)',
               'SPKL (test)',  'JSD (test)',  r'$L^2$-clr (test)',  r'$L^2$ (test)',  r'TV (test)',  r'Riemannian (test)' ]

    P_train = np.exp( Y_train )
    _scores = [ sym_perspective_kl( X_train, P_train ),
                js( X_train, P_train ),
                l2( X_train, P_train ),
                simplex_l2( X_train, P_train ),
                tv( X_train, P_train ),
                riemannian( X_train, P_train ) ]

    if Y_test is None:
        _scores += [ 0, 0, 0, 0, 0, 0 ]
    else:
        P_test   = np.exp( Y_test  )
        _scores += [ sym_perspective_kl( X_test, P_test ),
                     js( X_test, P_test ),
                     l2( X_test, P_test ),
                     simplex_l2( X_test, P_test ),
                     tv( X_test, P_test ),
                     riemannian( X_test, P_test ) ]

    return _scores, _names

