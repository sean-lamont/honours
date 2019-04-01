#!/usr/bin/env python 

'''
make the benchmark figures (fig.2 in the paper) based on given npz files
'''

import numpy as np
import os, sys

import matplotlib
matplotlib.use( 'Agg' )
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

def show_curves( ax, scores, idx, xlabel, ylabel, title, eps=1e-15 ):

    dim_range = range( 1, 11 )

    improve = 100 * ( 1-scores[1,:,idx]/(scores[0,:,idx] + eps) )
    for _x, _y, _text in zip( dim_range, scores[0,:,idx], improve ):
        if _y/scores[0,0,idx] < 1e-2:   # already 0
            pass
        elif _text>0:
            ax.text( _x+0.3, _y, '{0:.1f}%'.format(_text), fontsize=6, color='g', alpha=0.9 )
        else:
            ax.text( _x+0.3, _y, '{0:.1f}%'.format(_text), fontsize=6, color='r', alpha=0.9 )

    plt.plot( dim_range, scores[0,:,idx], label='clr-PCA',  color='k',    ls='-',  marker='d', lw=0.5 )
    plt.plot( dim_range, scores[1,:,idx], label='CoDA-PCA', color='g',    ls='-.', marker='v', alpha=0.6 )
    plt.plot( dim_range, scores[2,:,idx], label='SCoDA-PCA',color='b',    ls='--', marker='o', alpha=0.6 )
    plt.plot( dim_range, scores[3,:,idx], label='clr-AE',   color='r',    ls='-',  marker='*', alpha=0.6 )
    plt.plot( dim_range, scores[4,:,idx], label='CoDA-AE',  color='gold', ls=':',  marker='>', alpha=0.6 )
 
    if np.abs( scores[5,:,idx] ).max() > 1e-2:
        plt.plot( dim_range, scores[5,:,idx], label=r'CoDA-PCA$^\star$', color='g',  ls='-.',  marker='p', alpha=0.6 )

    plt.xticks( [dim_range[0], dim_range[-1]] )
    ax.set_xlabel( xlabel, labelpad=-12 )
    ax.set_ylabel( ylabel, labelpad=-12 )
    ax.set_title( title )

def main( filename1, filename2, mode=0 ):
    raw = np.load( filename1 )
    scores1 = raw['scores']
    measure_names = raw['measure_names']

    if filename2 is not None: scores2 = np.load( filename2 )['scores']

    if mode == 0:
        # 4 columns, 6 rows
        fig = plt.figure( figsize=(12,18), dpi=300 )
        for row in range(6):
            show_curves( fig.add_subplot( 6, 4, row*4+1 ), scores1, row,   r'\#PCs', '', measure_names[row] )
            show_curves( fig.add_subplot( 6, 4, row*4+2 ), scores1, row+6, r'\#PCs', '', measure_names[row+6] )
            show_curves( fig.add_subplot( 6, 4, row*4+3 ), scores2, row,   r'\#PCs', '', measure_names[row] )
            show_curves( fig.add_subplot( 6, 4, row*4+4 ), scores2, row+6, r'\#PCs', '', measure_names[row+6] )

        plt.savefig( 'fullcurves.pdf', bbox_inches='tight', pad_inches=0, transparent=True )

    elif mode == 1:
        # 2 rows, 3 columns
        fig = plt.figure( figsize=(9,6), dpi=300 )

        show_curves( fig.add_subplot( 2, 3, 1 ), scores1, 8,   r'\#PCs', '', measure_names[8]  )
        show_curves( fig.add_subplot( 2, 3, 2 ), scores1, 7,   r'\#PCs', '', measure_names[7]  )
        show_curves( fig.add_subplot( 2, 3, 3 ), scores1, 10,  r'\#PCs', '', measure_names[10] )
        show_curves( fig.add_subplot( 2, 3, 4 ), scores2, 8,   r'\#PCs', '', measure_names[8]  )
        show_curves( fig.add_subplot( 2, 3, 5 ), scores2, 7,   r'\#PCs', '', measure_names[7]  )
        show_curves( fig.add_subplot( 2, 3, 6 ), scores2, 10,  r'\#PCs', '', measure_names[10] )

        plt.savefig( 'curves.pdf', bbox_inches='tight', pad_inches=0, transparent=True )

    else:
        # high resolution small figures
        def printfig( filename, idx ):
            fig = plt.figure( figsize=(3,3), dpi=600 )
            show_curves( fig.add_subplot( 111 ), scores1, idx,   r'\#PCs', '', measure_names[idx] )
            plt.savefig( filename + ' ' + measure_names[idx] + '.pdf', bbox_inches='tight', pad_inches=0, transparent=True )

        for idx in range( 12 ):
            printfig( os.path.splitext( filename1 )[0], idx )

if __name__ == '__main__':
    if len( sys.argv ) < 2:
        print( "usage: {} results_dataset1.npz [results_dataset2.npz]".format(sys.argv[0]) )

    elif len( sys.argv ) == 2:
        main( sys.argv[1], None, 2 )

    else:
        main( sys.argv[1], sys.argv[2], 0 )

