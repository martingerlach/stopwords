import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## custom packages
src_dir = os.path.join('src')
sys.path.append(src_dir)
from filter_words import make_stopwords_filter
from scipy.stats import spearmanr


def calculate_jaccard_similarity(list_1_raw, list_2_raw):
    
    list_1 = list(set(list_1_raw))
    list_2 = list(set(list_2_raw))

    list_1_len = len(list_1)
    list_2_len = len(list_2)

    union_len = len(set(list_1+list_2))

    jaccard_similarity = (list_1_len + list_2_len - union_len)/ union_len

    return jaccard_similarity

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


list_S = ['INFOR','TFIDF','BOTTOM','TOP','MANUAL']

### SPEARMAN

list_S_val = []

df_S = pd.DataFrame(index=df.index,columns = list_S)

for i_S, str_S in enumerate(list_S):
    method = str_S
    cutoff_type = 'p'
    cutoff_val = 1.1 ## get all words
    df_filter = make_stopwords_filter(df,method=method,cutoff_type=cutoff_type,cutoff_val = cutoff_val)
    if method=='MANUAL':
        S_tmp = df['manual'].replace(to_replace=1.0,value=0).replace(to_replace=np.nan,value=1)
    else:
        S_tmp = df_filter['S']
    df_S[str_S] = S_tmp

N_S = len(list_S)
arr_C_rank = np.zeros((N_S-1,N_S-1))
for i1,S1 in enumerate(list_S[1:]):
    for i2,S2 in enumerate(list_S[:-1]):
        if i1>=i2:
            x1 = np.array(df_S[S1])
            x2 = np.array(df_S[S2])

            C = spearmanr(x1,x2)[0]
            
        else:
            C = np.nan
        arr_C_rank[i1,i2] = C


## JACCARD INDEX

list_cutoff = [('t',0.1),('t',9),('t',5),('n',1000),('n',len(df))]
list_S_w_sel = []

df_S = pd.DataFrame(index=df.index,columns = list_S)

for i_S, str_S in enumerate(list_S):
    method = str_S
    cutoff_type_val =  list_cutoff[i_S]
    cutoff_type = cutoff_type_val[0]
    cutoff_val = cutoff_type_val[1] 
    df_filter = make_stopwords_filter(df,method=method,cutoff_type=cutoff_type,cutoff_val = cutoff_val)
    w_sel = list(df_filter.index)
    list_S_w_sel += [w_sel]

N_S = len(list_S)
arr_C_jaccard = np.zeros((N_S-1,N_S-1))
for i1,S1 in enumerate(list_S[1:]):
    for i2,S2 in enumerate(list_S[:-1]):
        if i1>=i2:
            i1_sel = list_S.index(S1)
            x1 = list_S_w_sel[i1_sel]
            i2_sel = list_S.index(S2)
            x2 = list_S_w_sel[i2]
            try:
                C = calculate_jaccard_similarity(x1,x2)
            except ZeroDivisionError:
                C = 0
            print(S1,S2,C)
        else:
            C = np.nan
        arr_C_jaccard[i1,i2] = C




##################
### PLOT THE DATA
##################
x_label_list = ['INFOR', 'TFIDF', 'BOTTOM', 'TOP',]
y_label_list = ['TFIDF', 'BOTTOM', 'TOP', 'MANUAL']


# ## set colors
# cmap = 'tab10'
# ccolors = plt.get_cmap(cmap)(np.arange(10, dtype=int))

# set_blue = ccolors[0]
# set_orange = ccolors[1]
# set_green = ccolors[2]
# set_gray = ccolors[7]


# # ## set axis limits
# set_x_min = -.5
# set_x_max = 5.5
# set_y_min = -1
# set_y_max = 7


# ## make figure
# c1 = ccolors[4]
# c2 = ccolors[8]

label_rot = 30

fig, axes = plt.subplots(nrows=1, ncols=2 ,figsize=fig_size)
ax1 = axes[0]
im1 = ax1.imshow(arr_C_rank , vmin=-1, vmax=1, cmap='RdYlGn')

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.set_xticks(np.arange(len(x_label_list)))
ax1.set_yticks(np.arange(len(y_label_list)))
ax1.set_xticklabels(x_label_list)
ax1.set_yticklabels(y_label_list)
# ax1.set_xlabel('Spearman\'s rank correlation', fontsize=12)
ax1.set_title('Rank correlation')

for i in range(len(x_label_list)):
    for j in range(len(y_label_list)):
        if i < j:
            continue
        set_color = 'k'
        if abs(arr_C_rank[i, j]) > .7:
            set_color = 'w'
        text = ax1.text(j, i, "%.2f" % arr_C_rank[i, j],
                       ha="center", va="center", color=set_color, fontweight='bold')

plt.setp(ax1.get_xticklabels(), rotation=label_rot, ha="right", rotation_mode="anchor")






ax2 = axes[1]
im2 = ax2.imshow(arr_C_jaccard, vmin=-1, vmax=1, cmap='RdYlGn')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.set_xticks(np.arange(len(x_label_list)))
ax2.set_yticks(np.arange(len(y_label_list)))

ax2.set_xticklabels(x_label_list)
ax2.set_yticklabels([])
# ax2.set_xlabel('Jaccard index', fontsize=12)
ax2.set_title('Jaccard index')


for i in range(len(x_label_list)):
    for j in range(len(y_label_list)):
        if i < j:
            continue
        set_color = 'k'
        if abs(arr_C_jaccard[i, j]) > .7:
            set_color = 'w'
        text = ax2.text(j, i, "%.2f" % arr_C_jaccard[i, j],
                       ha="center", va="center", color=set_color, fontweight='bold')

plt.setp(ax2.get_xticklabels(), rotation=label_rot, ha="right", rotation_mode="anchor")

x_annot = -0.5
y_annot = 1.075
ax1.annotate(r'\textbf{C}',xy=(x_annot,y_annot),xycoords = 'axes fraction')


# Create colorbar
# cbar = ax.figure.colorbar(im1, ax=ax2)

cb_ax = fig.add_axes([.93, 0.25, 0.02, 0.5])
fig.colorbar(im2, cax=cb_ax, ticks=[-1, -.5, 0, .5, 1])


plt.subplots_adjust(wspace=.2,hspace=0.4)

# plt.show()
# plt.close()

filename_save = os.path.join(os.pardir,'results','figure-02c.png')
plt.savefig(filename_save,dpi=300,bbox_inches='tight')
