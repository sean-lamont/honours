#!/usr/bin/env python

'''
Run experiments on different datasets
'''

from __future__ import absolute_import, print_function, division

from CodaPCA import Alg, PCA, TSNE, CLRPCA, CodaPCA, NonParametricCodaPCA, clr_transform
from measure import compute_scores

import numpy as np
import os, sys, re, argparse, itertools

import matplotlib
matplotlib.use( 'Agg' )

from distutils.spawn import find_executable
if find_executable( 'latex' ):
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['text.latex.unicode'] = True

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Circle, Polygon

def read_csv( filename, spliter=',', normalize=True, min_val=0, dtype=np.float32 ):
    '''
    read CSV file
    the format of the CSV file is based on Aitchison's datasets
    '''

    armpattern = re.compile( r'^(\d+)arms$' )

    if armpattern.match( filename ):
        dim = int( armpattern.match( filename ).group(1) )
        return arms( dim ), None

    elif filename == 'digits':
        return digits(), None

    elif filename == 'spiral':
        return spiral( 2000, 7 ), None

    elif filename == 'clique':
        return clique( 2000, 7 ), None

    elif filename == 'blobs':
        return blobs( 2000, 7 ), None

    elif not os.access( filename, os.R_OK ):
        print( 'unable to read "{}", skipping'.format( filename ) )
        return None, None

    else:
        print( 'loading {}...'.format( filename ) )

    with open( filename ) as _file:
        def next_non_empty_line():
            while True:
                line = _file.readline()
                if line == '' or line.strip() != '': break
            return line.strip()

        # parse the header in first line
        line = next_non_empty_line()
        features = line.split( ',' )

        # parse a data line (see which field is float)
        pattern = re.compile( r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$' )
        line = next_non_empty_line()
        floatflag = [ (pattern.match(f) is not None) for f in line.split(spliter) ]

        features = list( itertools.compress( features, floatflag ) )
        arr = [ [ float(f) for f in itertools.compress( line.split(spliter), floatflag ) ] ]

        for line in _file:
            pureline = line.strip()
            if pureline == '': continue

            try:
                arr.append( [ float(f) for f in itertools.compress( pureline.split(spliter), floatflag ) ] )
            except ValueError as e:
                print( e )
                print( 'error parsing line {}'.format( pureline ) )

    arr = np.array( arr, dtype=dtype )
    if normalize: arr /= arr.sum( 1, keepdims=True )
    arr = np.maximum( arr, min_val )

    print( '{0} samples {1} features'.format( arr.shape[0], arr.shape[1] ) )
    print( 'sparsity: {}%'.format( 100 * ( arr < 1e-5 ).sum() / arr.size ) )

    return arr, features

def arms( dim, M=100 ):
    '''
    toy dataset on the simplex

    dim arms
    each arm has M samples
    '''
    print( 'generating {}arms dataset...'.format( dim ) )
    O = np.ones( dim ) / dim

    X = []
    for i in range( dim ):
        P = np.ones( dim ) * 1e-4
        P[i] = 1
        P /= P.sum()

        X += [ ( (1-lam) * O + lam * P ) for lam in np.linspace(0,1,M) ]

    X = np.array( X )
    print( '{0} samples {1} features'.format( X.shape[0], X.shape[1] ) )
    return X

def clique( N, k, seed=2019, dim=10 ):
    '''
        N -- sample size
        k -- size of clique
     seed -- random seed
      dim -- dimension of the probability vector

    return a list of Multinomail distributions and their labels
    '''
    assert( k > 0 )
    rng = np.random.RandomState( seed )

    M = N / ( (dim+1)*k )
    center = [ rng.dirichlet( np.ones(dim+1) ) for _ in range(k) ]

    theta = []
    #for i, j in itertools.product( range(k), range(k) ):
    #    if i < j:
    #        lam = np.linspace( 0,1,M )[:,None]
    #        theta.append( (1-lam) * center[i] + lam * center[j] )

    for i, j in itertools.product( range(dim+1), range(k) ):
        vertex = np.zeros( dim+1 )
        vertex[i] = 1
        lam = np.linspace( 0,1,M )[:,None]
        theta.append( (1-lam) * vertex + lam * center[j] )

    theta = np.vstack( theta )
    theta = np.maximum( theta, 1e-4 )
    theta /= theta.sum(1)[:,None]
    return theta

def spiral( N, k, seed=2019, dim=10 ):
    assert( k > 0 )
    rng = np.random.RandomState( seed )

    theta  = []
    labels = []

    cluster_size = [ int(N/k) for _ in range(k-1) ]
    cluster_size.append( N - sum(cluster_size) )

    for i, _n in enumerate( cluster_size ):
        vertex = np.zeros( dim+1 )
        vertex[i] = 1
        lam = (i +.5) / k
        center  = (1-lam)*vertex + lam*np.ones(dim+1)/(dim+1)

        noise = center.min() * ( 2 * rng.rand( _n, dim+1 ) - 1 )
        theta.append( center + noise )
        labels += [ i for _ in range(_n) ]

    theta = np.vstack( theta )
    theta = np.maximum( theta, 1e-4 )
    theta /= theta.sum(1)[:,None]
    return theta

def blobs( N, k, seed=2019, dim=10 ):
    '''
        N -- sample size
        k -- number of clusters
     seed -- random seed
      dim -- dimension of the probability vector

    return a list of Multinomail distributions and their labels
    '''
    assert( k > 0 )
    rng = np.random.RandomState( seed )

    theta  = []
    labels = []

    cluster_size = [ int(N/k) for _ in range(k-1) ]
    cluster_size.append( N - sum(cluster_size) )

    for i, _n in enumerate( cluster_size ):
        center = rng.dirichlet( np.ones(dim+1) )
        noise = center.min() * ( 2 * rng.rand( _n, dim+1 ) - 1 )
        theta.append( center + noise )
        labels += [ i for _ in range(_n) ]

    theta = np.vstack( theta )
    theta = np.maximum( theta, 1e-4 )
    theta /= theta.sum(1)[:,None]
    return theta

def digits():
    '''
    Note that digits is not compositional data
    '''
    import sklearn.datasets
    X = sklearn.datasets.load_digits()
    X = np.array( X.data, dtype=np.float32 )
    X += 1e-3
    X /= X.sum( 1, keepdims=True )

    print( '{0} samples {1} features'.format( X.shape[0], X.shape[1] ) )
    return X

def read_factors( filename, spliter=',' ):
    '''
    factors (labels) CSV file
    '''
    arr = []
    num_factors = None

    with open( filename ) as _file:
        next( _file ) # skip the first line, which is the header

        for line in _file:
            if line.strip() == '': continue

            fields = line.strip().split( spliter )

            if num_factors is None:
                num_factors = len( fields )
            else:
                if len( fields ) != num_factors:
                    raise RuntimeError( 'error parsing line {}'.format( line ) )

            arr.append( fields )

    return arr

def parse_color_marker( factor_filename, c, m ):
    '''
    generate colors and markers based on a given factor file

    factor_filename -- factor filename
                  c -- show which column with color
                  m -- show which column with marker
    '''
    L = read_factors( factor_filename )

    def get_map( idx ):
        _count = {}
        for _labels in L:
            if _labels[idx] in _count:
                _count[_labels[idx]] += 1
            else:
                _count[_labels[idx]] = 1

        _list = [ (name,_count[name]) for name in _count ]
        _list.sort( key=lambda _: _[1], reverse=True )

        return { c[0]:i for i,c in enumerate(_list) }, _list

    color_map, color_list = get_map( c )
    C = np.array( [ color_map[_labels[c]] for _labels in L ], dtype=np.float32 )
    y = np.array( [ color_map[_labels[c]] for _labels in L ], dtype=np.int )

    Cnames = [ (i/C.max(), '{} ({})'.format(name, count)) for i,(name,count) in enumerate(color_list) ]
    C /= C.max()

    all_markers = [ '.', '*', '>', 'v', 's', '8', 'p', 'x', '<', 'd' ]
    marker_map, marker_list = get_map( m )
    M = [ all_markers[ marker_map[_labels[m]] ] for _labels in L ]
    Mnames = [ (all_markers[i], '{0}({1})'.format(name, count)) for i,(name,count) in enumerate(marker_list) ]

    return C, Cnames, M, Mnames, y

def show_scatter( ax, title, A, A_axis, mask, features, args ):
    '''
    show the scatter plot (mainly A matrix)
    '''
    if args.factor is not None:
        cmap = matplotlib.cm.get_cmap( 'rainbow' )

        # display index of each point in gray style
        # for i, (x, y) in enumerate( A ):
            # ax.text( x, y, str(i+1),
                     # color='gray',
                     # fontsize=4,
                     # alpha=0.4,
                     # horizontalalignment='center',
                     # verticalalignment='center' )

        # scatter with markers/colors
        C, Cnames, M, Mnames, y = parse_color_marker( args.factor, args.color, args.marker )
        if A.shape[1] > 2:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis(n_components=2)
            A   = lda.fit_transform( A, y )
            print( lda.explained_variance_ratio_, lda.explained_variance_ratio_.sum() )

        for marker in set( M ):
            idx = np.array( [ _m==marker for _m in M ] )
            ax.scatter( A[idx,0], A[idx,1], c=C[idx], marker=marker, cmap=cmap, alpha=.5, s=7 )

        # generate the legend
        handles = []
        for c, name in Cnames:
            handles.append( mpatches.Patch(color=cmap(c), label=name ) )
        for m, name in Mnames:
            handles.append( mlines.Line2D( [], [], c='k', lw=0.5, marker=m, label=name ) )
        #ax.legend( handles=handles, loc='best', prop={'size': 6} )

    else:
        cmap = matplotlib.cm.get_cmap( 'Spectral' )
        for i, (x, y) in enumerate( A ):
            if not mask[i]:
                ax.text( x, y, str(i+1),
                         color=cmap(i/A.shape[0]),
                         fontsize=4,
                         fontweight='black',
                         alpha=0.8,
                         horizontalalignment='center',
                         verticalalignment='center' )

        ax.scatter( A[mask,0], A[mask,1], color='black', s=8, alpha=0.8 )

    # show the axis
    if A_axis is not None:
        cmap = matplotlib.cm.get_cmap( 'ocean' )
        score_a = []
        for i, a in enumerate( axis ):
            _xy = axis_curves[i*20:(i+1)*20] - A.mean(0)
            score_a.append( (a, np.linalg.norm(_xy) ) )
        score_a.sort( key=lambda _ : _[1], reverse=True )
        a_list = [ score_a[i][0] for i in range(7) ]

        c = 0
        for i, a in enumerate( axis ):
            if not (a in a_list): continue

            color = c / ( len(a_list)+1 )
            c += 1

            _x = axis_curves[i*20:(i+1)*20,0]
            _y = axis_curves[i*20:(i+1)*20,1]

            ax.plot( _x, _y, c=cmap(color) )
            ax.arrow( _x[-2], _y[-2], _x[-1]-_x[-2], _y[-1]-_y[-2], color=cmap(color), head_width=.3 )
            ax.text(  _x[-1], _y[-1], features[a], fontsize=7, color=cmap(color), alpha=.7 )
            print( features[a] )

    # set the plotting axis
    xmin = A[:,0].min()
    xmax = A[:,0].max()
    margin = 0.05 * (xmax-xmin)
    xmin -= margin
    xmax += margin
    ax.set_xlim( xmin, xmax )
    ax.set_xticks( [xmin, xmax] )

    ymin = A[:,1].min()
    ymax = A[:,1].max()
    margin = 0.05 * (ymax-ymin)
    ymin -= margin
    ymax += margin
    ax.set_ylim( ymin, ymax )
    ax.set_yticks( [ymin, ymax] )

    ax.set_xlabel( r'\#comp1', labelpad=-10 )
    ax.set_ylabel( r'\#comp2', labelpad=-25 )
    ax.set_title( title )

def construct_axis( X, points=20 ):
    '''
    construct a set of axis in the simplex space
    '''
    X_test = []
    O = X.mean( 0 )

    for a in range( X.shape[1] ):
        tmin = ( X[:,a].min() - O[a] ) / (1-O[a])
        tmax = ( X[:,a].max() - O[a] ) / (1-O[a])

        V = np.zeros( X.shape[1] )
        V[a] = 1

        X_test.append( np.array( [ ((1-t)*O + t*V) for t in np.linspace(tmin,tmax,points) ] ) )

    return np.vstack( X_test )

def visualize( args, d=2, repeat=1, mode=1 ):
    '''
    visualize CoDA datasets in 2D plots

    d      -- target dimensionality
    repeat -- number of repeated runs per algorithm
    mode   -- format of figure
    '''

    X, features = read_csv( args.csv_file )
    N = X.shape[0]
    if X is None: return

    print( 'generating scatter plots...' )

    def visualize_alg( ax, title, alg ):
        mask = np.zeros( N, dtype=np.bool )

        if alg == Alg.PCA:
            pca = PCA( d )
            pca.fit( X )
            A = pca.transform( X )

            # PCA reconstruct may go outside the simplex
            mask = np.any( pca.project( X )<0, axis=1 )

        elif alg == Alg.CLRPCA:
            pca = CLRPCA( d )
            pca.fit( X, repeat=repeat, verbose=False )
            A = pca.transform( X )

        elif alg in [ Alg.CODAPCA, Alg.SCODAPCA, Alg.CLRAE, Alg.CODAAE ]:
            pca = CodaPCA( d,
                           args.lrate,
                           args.nn_shape,
                           batchsize=args.batchsize,
                           alg=alg )
            pca.fit( X, epochs=args.epochs, repeat=repeat, verbose=True )
            A = pca.transform( X )

        elif alg == Alg.NONPARACODAPCA:
            pca = NonParametricCodaPCA( d )
            A   = pca.fit_transform( X )

        elif alg == Alg.TSNE:
            pca = TSNE( 2 )
            A   = pca.fit_transform( X )

        else:
            raise RuntimeError( 'unknown alg' )

        if args.axis:
            A_axis = pca.transform( construct_axis( X ) )
        else:
            A_axis = None

        # plot the 2D scatter
        show_scatter( ax, title, A, A_axis, mask, features, args )
        print( '{} done'.format( title ) )

    if mode == 0:
        # make a one-row figure
        fig = plt.figure( figsize=(15,3), dpi=300 )

        visualize_alg( fig.add_subplot( 151 ), 'PCA',      Alg.PCA )
        visualize_alg( fig.add_subplot( 152 ), 'clr-PCA',  Alg.CLRPCA )
        visualize_alg( fig.add_subplot( 153 ), 'CoDA-PCA', Alg.CODAPCA )
        #visualize_alg( fig.add_subplot( 153 ), 'CoDA-PCA', Alg.NONPARACODAPCA ) # alternatively, use the l-BFGS optimizer
        visualize_alg( fig.add_subplot( 154 ), 'CoDA-AE',  Alg.CODAAE )
        visualize_alg( fig.add_subplot( 155 ), 't-SNE',    Alg.TSNE )

        ofilename = os.path.splitext( args.csv_file )[0] + "_scatter.pdf"
        plt.savefig( ofilename, bbox_inches='tight', pad_inches=0, transparent=True )
        print( 'figure saved to {}'.format( ofilename ) )

    elif mode == 1:
        # include more algorithms in a two-row figure
        fig = plt.figure( figsize=(10,7), dpi=300 )

        visualize_alg( fig.add_subplot( 231 ), 'CLR-PCA',   Alg.CLRPCA )
        visualize_alg( fig.add_subplot( 232 ), 'CoDA-PCA',  Alg.CODAPCA )
        visualize_alg( fig.add_subplot( 233 ), 'SCoDA-PCA', Alg.SCODAPCA )
        visualize_alg( fig.add_subplot( 234 ), 'CLR-AE',    Alg.CLRAE )
        visualize_alg( fig.add_subplot( 235 ), 'CoDA-AE',   Alg.CODAAE )
        visualize_alg( fig.add_subplot( 236 ), 't-SNE',     Alg.TSNE )

        ofilename = os.path.splitext( args.csv_file )[0] + "_scatter.pdf"
        plt.savefig( ofilename, bbox_inches='tight', pad_inches=0, transparent=True )
        print( 'figure saved to {}'.format( ofilename ) )

    else:
        # high resolution separated figures
        def unifigure( name, alg ):
            # include more algorithms in a two-row figure
            fig = plt.figure( figsize=(3,3), dpi=600 )
            visualize_alg( fig.add_subplot( 111 ), name, alg )
            plt.savefig( name + ".eps", bbox_inches='tight', pad_inches=0, transparent=True )

        unifigure( 'PCA',       Alg.PCA )
        unifigure( 'CLR-PCA',   Alg.CLRPCA )
        unifigure( 'CoDA-PCA',  Alg.CODAPCA )
        unifigure( 'SCoDA-PCA', Alg.SCODAPCA )
        unifigure( 'CLR-AE',    Alg.CLRAE )
        unifigure( 'CoDA-AE',   Alg.CODAAE )
        unifigure( 't-SNE',     Alg.TSNE )

def benchmark( args ):
    '''
    benchmark experiments
    '''
    X, _tmp = read_csv( args.csv_file )
    if X is None: return

    dim_range = range( 1, min( X.shape[1], args.max_dim+1 ) )

    ofilename = args_string( args ) + '.npz'
    if os.access( ofilename, os.R_OK ):
        print( "file '{}' already exists, delete it first".format( ofilename ) )
        return

    print( "benchmarking (this may take quite a while)..." )
    train, test = train_test_split( X.shape[0], args.test_ratio )
    scores = np.zeros( [6, len(dim_range), 12] )

    def benchmark_codapca( n_components, alg ):
        if alg == Alg.CLRPCA:
            pca = CLRPCA( n_components )
            pca.fit( X[train], verbose=False )
            Y_train = pca.project( X[train] )
            Y_test  = pca.project( X[test]  )

        elif alg == Alg.NONPARACODAPCA:
            pca = NonParametricCodaPCA( n_components )
            Y_train = pca.project( X[train] )
            Y_test  = None

        elif alg in [ Alg.CLRPCANN, Alg.CODAPCA, Alg.SCODAPCA, Alg.CLRAE, Alg.CODAAE ]:
            pca = CodaPCA( n_components,
                           args.lrate,
                           args.nn_shape,
                           batchsize=args.batchsize,
                           alg=alg )
            pca.fit( X[train], epochs=args.epochs, verbose=False )
            Y_train = pca.project( X[train] )
            Y_test  = pca.project( X[test]  )

        else:
            raise RuntimeError( 'unknown alg' )

        _ret = compute_scores( X[train], Y_train, X[test], Y_test )
        if ( np.isnan(_ret[0]).sum() > 0 ): print( alg, 'produced NAN scores. BAD' )
        return _ret

    for i, n_components in enumerate( dim_range ):
        print( "#PC={}".format( n_components ) )

        _scores, measure_names = benchmark_codapca( n_components, Alg.CLRPCA )
        scores[0,i,:] = _scores

        scores[1,i,:] = benchmark_codapca( n_components, Alg.CODAPCA )[0]
        scores[2,i,:] = benchmark_codapca( n_components, Alg.SCODAPCA )[0]
        scores[3,i,:] = benchmark_codapca( n_components, Alg.CLRAE )[0]
        scores[4,i,:] = benchmark_codapca( n_components, Alg.CODAAE )[0]
        scores[5,i,:] = benchmark_codapca( n_components, Alg.NONPARACODAPCA )[0]

    print( "done" )
    np.savez( ofilename, scores=scores, measure_names=measure_names )

def train_test_split( N, test_ratio ):
    '''
    random split into training/testing sets
    '''
    allidx = np.arange( N )
    np.random.shuffle( allidx )

    N_train = int( N * (1-test_ratio) )
    return allidx[:N_train], allidx[N_train:]

def args_string( args ):
    '''
    generate a string based on given args (for file name)
    '''

    name = '{}_lrate{}_batchsize{}_epochs{}_maxdim{}_test{}'.format(
            args.csv_file,
            args.lrate,
            args.batchsize,
            args.epochs,
            args.max_dim,
            args.test_ratio )

    return name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'csv_file',  type=str, help='csv file' )

    parser.add_argument( '--visualize', action='store_true', help='perform visualization (otherwise benchmarking)' )
    parser.add_argument( '--factor',    type=str, help='factor csv file' )
    parser.add_argument( '--color',     type=int, help='which column of profile for coloring' )
    parser.add_argument( '--marker',    type=int, help='which column of profile for showing markers' )
    parser.add_argument( '--axis',      action='store_true', help='show axis' )

    parser.add_argument( '--lrate',      type=float, default=1e-3, help='learning rate' )
    parser.add_argument( '--nn_shape',   type=int,   nargs='+', default=[100,100], help='shape of the parametric mapping into latent space' )
    parser.add_argument( '--batchsize',  type=int,   default=32,  help='size of a mini-batch' )
    parser.add_argument( '--epochs',     type=int,   default=300, help='number of epochs' )
    parser.add_argument( '--max_dim',    type=int,   default=10,  help='maximum reduced dimensionality for benchmarking' )
    parser.add_argument( '--test_ratio', type=float, default=0.1, help='ratio of testing samples' )

    args = parser.parse_args()
    _vis = args.visualize
    del args.visualize
    for opt in vars(args):
        print( '{0:20} : {1}'.format( opt, getattr(args, opt) ) )

    np.random.seed( 2019 )

    if _vis:
        visualize( args )
    else:
        benchmark( args )

if __name__ == '__main__':
    main()
