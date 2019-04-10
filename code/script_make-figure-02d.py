import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## custom packages
src_dir = os.path.join('src')
sys.path.append(src_dir)
from filter_words import make_stopwords_filter


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
          'legend.fontsize': 6,
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


## setup binning

max_freq = df['N'].max()
min_freq = df['N'].min()

base_set = 10
bin_start = int(np.log(min_freq * .9)/ np.log(base_set)) 
bin_end = int(np.log(max_freq * 1.1)/ np.log(base_set)) 
bin_num = bin_end - bin_start + 1

bin_list = np.logspace(bin_start, bin_end, num = bin_num, base=base_set)

## select some stopword lists
list_S = ['INFOR','TFIDF','MANUAL']
list_cutoff = [('t',0.1),('t',9),('n',len(df))]
list_p_w_bin = []

N_w_bin_all,edges = np.histogram(df['N'],bins=bin_list,)


for i_S, str_S in enumerate(list_S):
    method = str_S
    cutoff_type_val =  list_cutoff[i_S]
    cutoff_type = cutoff_type_val[0]
    cutoff_val = cutoff_type_val[1] 
    df_filter = make_stopwords_filter(df,method=method,cutoff_type=cutoff_type,cutoff_val = cutoff_val)
    N_sel = np.array(df['N'].loc[df_filter.index])
    
    
    N_w_bin_tmp,edges = np.histogram(N_sel,bins=bin_list,)
    p_w_bin_tmp = N_w_bin_tmp/N_w_bin_all
    list_p_w_bin += [p_w_bin_tmp]



## PLOTTING
cmap = 'tab10'
ccolors = plt.get_cmap(cmap)(np.arange(10, dtype=int))


color_dict = {
    'INFOR': ccolors[0], 
    'TFIDF': ccolors[1],  
    'MANUAL': ccolors[2],  
#     'Top': ccolors[3], 
#     'Mallet': ccolors[2], 
#     'Rand': ccolors[7], 
}

x_label_list = [ r'$10^{%s}$'%(int(np.log10(h))) for h in  bin_list]

fig, ax = plt.subplots( figsize=fig_size)


width=1/8
fix_shift = width/2


for i_S, str_S in enumerate(list_S):
    y = list_p_w_bin[i_S]
    x = np.arange(len(bin_list[:-1]))


    ax.bar(x + fix_shift + width * (i_S+1), y, width=width, color=color_dict[str_S], label=list_S[i_S] )
    


ax.set_xticks(np.arange(len(bin_list)))
ax.set_xticklabels(x_label_list, visible=1)

ax.set_xlabel('Word frequency $n(w)$')
ax.set_ylabel('Fractoin of stopwords')
ax.set_yscale('log')
ax.legend(loc='upper center',frameon=False, ncol=3,)

ax.set_ylim(0.9*10**(-3),2*1)

x_annot = -0.2
y_annot = 1.05
ax.annotate(r'\textbf{D}',xy=(x_annot,y_annot),xycoords = 'axes fraction')



filename_save = os.path.join(os.pardir,'results','figure-02d.png')
plt.savefig(filename_save,dpi=300,bbox_inches='tight')
