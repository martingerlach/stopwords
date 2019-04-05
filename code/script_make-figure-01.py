import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###########
## Setup of panel##
###########
# number of pt for column in latex-document
fig_width_pt = 255  # single-column:510, double-column: 246; Get this from LaTeX using \showthe\columnwidth
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
          
#           'text.usetex': True,
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

##################
### READ THE DATA
##################
corpus_name = '20NewsGroup'
filename = os.path.join(os.pardir,'results','%s_stopword-statistics.csv'%(corpus_name))
df = pd.read_csv(filename,index_col=0,na_filter = False)
df['manual']=df['manual'].replace(to_replace='',value=np.nan).replace(to_replace='1.0',value=1.0)

arr_N = np.array(df['N'])
arr_H = np.array(df['H'])
arr_H_tilde = np.array(df['H-tilde'])
arr_H_tilde_std = np.array(df['H-tilde_std'])

list_w = list(df.index)

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


## set axis limits
x_annot1 = -0.2
y_annot1 = 1.06

x_annot2 = -0.1
y_annot2 = 1.06

x_annot_text = 0.05
y_annot_text = 0.95

ymin,ymax=-1,16.

y_lable_list = np.linspace(0, 15, 4).astype(int)


## make figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

# ax = axes[0, 0]

D = 18803 ## Number of documents to get maximum possible value for entropy
Dmax = np.log2(D)
ax.plot([np.min(arr_N),np.max(arr_N)], [Dmax,Dmax],  lw=1, ls=':',  color='black')

x = arr_N
y = arr_H
ax.plot(x, y, ms=.5, lw=0, marker='o', color=set_blue, alpha=0.5, rasterized=1,zorder=1)

ind_sort = np.argsort(arr_N)
x = arr_N[ind_sort]
y = arr_H_tilde[ind_sort]
yerr = arr_H_tilde_std[ind_sort]*5
ax.errorbar(x, y, yerr=yerr, color=set_orange, lw=1, alpha=0.9, rasterized=1,zorder=2)

list_w_sel = ['cancer','information', 'thanks', 'article', 'the'] ## words we mark
x_factor_list = [1., 1., 0.075, 0.075, 0.9,]
y_factor_list = [0.6, 0.8, 1., 1.0, 0.8,] ## where to put the text of the selected words
for i_sel, w_sel in enumerate(list_w_sel):
    iw = list_w.index(w_sel)
    x = arr_N[iw]
    y = arr_H[iw]
    ax.scatter(x, y, marker='x', facecolors=set_green,zorder=3)

    x = arr_N[iw]*x_factor_list[i_sel]
    y = arr_H[iw]*y_factor_list[i_sel]
    ax.text(x,y, w_sel, color=set_green)
    
ax.annotate(r'A',xy=(x_annot1,y_annot1),xycoords = 'axes fraction',fontweight='bold')
ax.annotate('English',xy=(x_annot_text,y_annot_text),xycoords = 'axes fraction')

ax.set_xscale("log")

ax.set_xlabel('Word frequency $n(w)$')
ax.set_ylabel('$H(w|C)$')

ax.set_yticks(y_lable_list)
ax.set_yticklabels(y_lable_list, visible=1)
# # ax1.set_ylim([-1, 16])
ax.set_ylim(ymin,ymax)


plt.subplots_adjust(left=0.16,bottom=0.2)

filename_save = os.path.join(os.pardir,'results','figure-01.png')
plt.savefig(filename_save,dpi=300)