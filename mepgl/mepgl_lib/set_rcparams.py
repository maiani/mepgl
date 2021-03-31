"""
Set rcparams for matplotlib plots.
"""

from matplotlib import pyplot as plt

params = {
   'font.family' : 'STIXGeneral',
   'mathtext.fontset': 'stix',
   'axes.labelsize': 10,
   'legend.fontsize': 9,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'figure.figsize': [3.4, 2.5]
   }


plt.rcParams.update(params)

plt.close('all')