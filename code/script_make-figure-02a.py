import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###########
## Setup ##
###########
# number of pt for column in latex-document
fig_width_pt = 510/2  # single-column:510, double-column: 246; Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.1/72.27 # Convert pt to inches
width_vs_height = (np.sqrt(5)-1.0)/1.75#(np.sqrt(5)-1.0)/2.0 # Ratio of height/width [(np.sqrt(5)-1.0)/2.0]
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = width_vs_height*fig_width  # height in inches
fig_size = [fig_width,fig_height]

# here you can set the parameters of the plot (fontsizes,...) in pt
params = {'backend': 'ps',
          'axes.titlesize':8,
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
#           'figtext.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          
          'text.usetex': True,
          'ps.usedistiller' : 'xpdf',
          'figure.figsize': fig_size,
          'text.latex.unicode':True,
          'text.latex.preamble': [r'\usepackage{bm}'],
          
          'xtick.direction':'out',
          'ytick.direction':'out',
          
          'axes.spines.right' : False,
          'axes.spines.top' : False
         }
plt.rcParams.update(params)

set_b = 0.22 # set bottom
set_l = 0.1 # set left
set_r = 0.925 # set right
set_hs = 0.2 # set horizontal space
set_vs = 0.25 # set vertical space

set_ms = 0.0 # set marker size
set_lw = 2.5 # set line width
set_alpha = 0.8

##################
### READ THE DATA
##################
corpus_name = '20NewsGroup'
filename = os.path.join(os.pardir,'results','%s_stopword-statistics.csv'%(corpus_name))
df = pd.read_csv(filename,index_col=0)

arr_N = np.array(df['N'])
arr_H = np.array(df['H'])
arr_H_tilde = np.array(df['H-tilde'])
arr_I = arr_H_tilde-arr_H

heatmap, xedges, yedges = np.histogram2d(np.log10(arr_N), arr_I, bins=50)
heatmap_log = np.log10(heatmap)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


##################
### PLOT THE DATA
##################

## set colors
cmap = 'tab10'
ccolors = plt.get_cmap(cmap)(np.arange(10, dtype=int))

set_blue = ccolors[0]
set_orange = ccolors[1]
set_green = ccolors[2]
set_gray = ccolors[7]


# ## set axis limits
set_x_min = -.5
set_x_max = 5.5
set_y_min = -1
set_y_max = 7


## make figure
fig = plt.figure(figsize=fig_size)

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)


t= np.arange(1000)/100.
x = np.sin(np.pi**t)
y = np.cos(np.pi**t)
z = np.cos(0.1*np.pi**t)


## subplot 1
bin_1 = 30
ax1.hist(np.log10(arr_N), bin_1, histtype='step')
ax1.set_yscale("log")
ax1.set_ylim(1, 60000)
ax1.set_xlim(set_x_min, set_x_max)
ax1.set_xticks([0, 1, 2, 3, 4, 5])
ax1.set_xticklabels([])
ax1.set_yticks([10**0,10**2,10**4])

ax1.set_ylabel('Count')

## subplot 2
cax = ax2.imshow(heatmap_log.T, extent=extent, origin='lower', cmap='gnuplot2_r', aspect="auto")
ax2.set_ylim(set_y_min, set_y_max)
ax2.set_ylabel('Information $I(w)$')
ax2.set_xlabel('Word frequency $n(w)$')
ax2.set_xlim(set_x_min, set_x_max)
ax2.set_xticks([0, 1, 2, 3, 4, 5])
ax2.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$',])
ax2.set_yticks([0,2,4,6])


# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(['$10^0$', '$10^1$','$10^2$','$10^3$','$10^4$'])  # horizontal colorbar
cbar.set_clim(-1.0, 6.0)

## subplot 3
bin_3 = 30
ax3.hist(arr_I, bin_3,orientation='horizontal', histtype='step' )
ax3.set_xscale("log")

ax3.set_xticks([10**0,10**2,10**4])

ax3.set_ylim(set_y_min, set_y_max)
ax3.set_yticklabels([])
ax3.set_xlabel('Count')

###########
# end
###########

x_annot = -0.3
y_annot = 1.05
ax1.annotate(r'\textbf{A}',xy=(x_annot,y_annot),xycoords = 'axes fraction')

# fig.patch.set_alpha(1.)
# ax1.patch.set_alpha(0.)

plt.subplots_adjust(wspace=.2, hspace=0.3)

# plt.show()
# plt.close()

filename_save = os.path.join(os.pardir,'results','figure-02a.png')
plt.savefig(filename_save,dpi=300,bbox_inches='tight')
