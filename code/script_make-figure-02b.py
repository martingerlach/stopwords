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
df = pd.read_csv(filename,index_col=0,na_filter = False)
df['manual']=df['manual'].replace(to_replace='',value=np.nan).replace(to_replace='1.0',value=1.0)

arr_F = np.array(df['F'])
arr_H = np.array(df['H'])
arr_H_tilde = np.array(df['H-tilde'])
arr_I = arr_H_tilde-arr_H

S = df['I']
S = S.dropna().sort_index().sort_values(kind='mergesort')
df_filter = pd.DataFrame(index = S.index )
df_filter['F-cumsum']=df.loc[S.index]['F'].cumsum()
df_filter['S'] = S



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
c1 = ccolors[4]
c2 = ccolors[8]

fig, ax = plt.subplots(figsize=fig_size)

# ax.plot(x_I_array, y_token_list, c=c1,  label='Word tokens')
# ax.plot(x_I_array, y_type_list, c=c2, label='Word types')


x = np.array(df_filter['S'])
y = 1-np.array(df_filter['F-cumsum'])
ax.plot(x, y, c=set_blue,  )
# ax.plot(x_I_array, y_type_list, c=c2, )


y_sel = 0.2
x_sel = df_filter[df_filter['F-cumsum']<1-y_sel]['S'].iloc[-1]

ax.plot([set_x_min,x_sel],[y_sel,y_sel], ':', color='r')
ax.plot([x_sel,x_sel],[set_y_min,y_sel], ':', color='r')

 
ax.set_xlim(-.1, 2)
ax.set_ylim(0, 1.1)
ax.patch.set_alpha(0.0)

# ax.set_xticks([0,1,2])
# ax.set_yticks([0,0.5,1])

ax.set_xticks([0,0.5,1,1.5,2])
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])


ax.set_xlabel('Information $I(w)$')
ax.set_ylabel('Survival function')

# ax.set_xlim(set_x_min, set_x_max)


# ax.legend(loc='upper right',frameon=False)





x_annot = -0.2
y_annot = 1.05
ax.annotate(r'\textbf{B}',xy=(x_annot,y_annot),xycoords = 'axes fraction')

plt.subplots_adjust(wspace=.2, hspace=0.3)

# plt.show()
# plt.close()

filename_save = os.path.join(os.pardir,'results','figure-02b.png')
plt.savefig(filename_save,dpi=300,bbox_inches='tight')
