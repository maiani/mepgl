"""
Set rcparams for matplotlib plots.
"""

import matplotlib.pyplot as plt
from math import sqrt

golden_ratio = (1 + sqrt(5))/2

figwidth = 3.4
fontsize = 10

rcParams = {
   'mathtext.fontset'           : 'stix',
   'font.family'                : 'STIXGeneral',
   'axes.labelsize'             : fontsize,
   'axes.titlesize'             : fontsize + 1,
   'xtick.labelsize'            : fontsize,
   'ytick.labelsize'            : fontsize, 
   'text.usetex'                : True,
   'figure.figsize'             : [figwidth, figwidth/golden_ratio],
   'axes.grid'                  : False,
   'legend.fancybox'            : False,
   'legend.edgecolor'           : 'black',
   'legend.framealpha'          : 1.0,
   'legend.fontsize'            : fontsize - 1,
   'figure.dpi'                 : 1000,
   'pdf.compression'            : 6,
   'lines.linewidth'            : 1,
   'lines.linestyle'            : '-',
   'lines.marker'               : '.',
   'lines.markersize'           : '5',
   }


plt.rcParams.update(rcParams)
plt.close('all')