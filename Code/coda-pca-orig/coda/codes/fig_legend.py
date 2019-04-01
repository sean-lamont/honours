#!/usr/bin/env python 

'''
create the legend figure in the NIPS paper
'''

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#matplotlib.use( 'Agg' )
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fig = plt.figure( figsize=(6,0.3), dpi=300 )

ms = 8
lines = [ mlines.Line2D( [], [], color='k',    marker='d', markersize=ms, label='clr-PCA'),
          mlines.Line2D( [], [], color='g',    marker='v', markersize=ms, label='CoDA-PCA'),
          mlines.Line2D( [], [], color='g',    marker='p', markersize=ms, label='CoDA-PCA$^*$'),
          mlines.Line2D( [], [], color='b',    marker='o', markersize=ms, label='SCoDA-PCA'),
          mlines.Line2D( [], [], color='r',    marker='*', markersize=ms, label='clr-AE'),
          mlines.Line2D( [], [], color='gold', marker='>', markersize=ms, label='CoDA-AE') ]
fig.legend( handles=lines, frameon=False, loc='center', ncol=len(lines) )

plt.savefig( 'legend.pdf', bbox_inches='tight', pad_inches=0, transparent=True )
