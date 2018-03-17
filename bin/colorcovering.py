# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

from itertools import combinations
from tqdm import tqdm
from scipy import stats, interpolate
from scipy.special import binom
from astropy.table import Table

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def calc(train, test, filename=None):
    train_table = Table.read(train, format='ascii')
    test_table = Table.read(test, format='ascii')
    
    train_cols = [c for c in train_table.colnames[1:-1] if not c.startswith('e')]
    test_cols = [c for c in test_table.colnames[1:] if not c.startswith('e')]

    eqcols = all([x==y for x,y in zip(train_cols, test_cols)])    
    if not eqcols :
        raise ValueError('Different colors in training and test samples!!!')

    train_colors = train_table[train_cols]
    n_combColors = binom(len(train_cols), 2)
    good_mask = np.full((len(test_table), int(n_combColors)), np.nan)
        
    m=0
    for c1,c2 in tqdm(combinations(train_cols, 2), total=n_combColors) :
        values = np.vstack([train_colors[c1], train_colors[c2]])
        kernel = stats.gaussian_kde(values)
        
        xmin = train_colors[c1].min()
        xmax = train_colors[c1].max()
        ymin = train_colors[c2].min()
        ymax = train_colors[c2].max()
               
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        Z = Z / Z.sum()
        
        t = np.linspace(0, Z.max(), 1000)
        mask = (Z >= t[:, None, None])
        integral = ( mask * Z).sum(axis=(1,2))

        f = interpolate.interp1d(integral, t)
        t_contours = f(np.array([0.90]))
        
        test_Z = interpolate.griddata(positions.T, Z.ravel(), 
                                      (test_table[c1], test_table[c2]), 
                                      fill_value=0)
        good_mask[:,m] = test_Z > t_contours[0]
        m += 1            
        
        if c1 == 'gmi' and c2 == 'rmz' :        
            idx_good = np.nonzero(test_Z > t_contours[0])
            
            fig, ax = plt.subplots()
        
            ax.contour(Z.T, t_contours, colors='k', 
                       extent=[xmin,xmax,ymin,ymax], 
                       zorder=1000)
        
            ax.plot(test_table[c1], test_table[c2], 
                    'bo', markersize=5, rasterized=True)
            ax.plot(test_table[idx_good][c1], test_table[idx_good][c2], 
                    'g^', markersize=5, rasterized=True)
            ax.plot(train_colors[c1], train_colors[c2], 
                    'k.', markersize=5, rasterized=True)
    
            meanc = np.mean(test_table[c1])
            stdc = np.std(test_table[c1])
            ax.set_xlim(meanc-3*stdc, meanc+3*stdc)
    
            meanc = np.mean(test_table[c2])
            stdc = np.std(test_table[c2])
            ax.set_ylim(meanc-3*stdc, meanc+3*stdc)
        
            ax.set_xlabel(c1.replace('m','-'), fontsize=24)
            ax.set_ylabel(c2.replace('m','-'), fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
                    
            fig.tight_layout()
            
            if filename is not None :
                root, ext = os.path.splitext(filename)
                filename1 = '{}_{}vs{}{}'.format(root, c1, c2, ext)
                plt.savefig(filename1)
                plt.close()
            else :
                plt.show()

    test_good_all = np.all(good_mask, axis=1)
    idx_good = np.nonzero(test_good_all)
    print(len(test_table[idx_good])/len(test_table))
    
    test_good_any = np.any(good_mask, axis=1)
    idx_agood = np.nonzero(test_good_any)
    print(len(test_table[idx_agood])/len(test_table))
    
    inTSCS = test_good_all, test_good_any
    inTSCS = Table(inTSCS, names=['inTSCS_ALL', 'inTSCS_ANY'])

    
    colors = test_table[train_cols].to_pandas()    
    sample_label = np.array(["c_bad" for x in range(len(test_good_all))])
    sample_label[idx_agood] = "b_any"
    sample_label[idx_good] = "a_all"
    colors = colors.assign(sample=pd.Categorical(sample_label))
    
    sns.set(font_scale=1.25)
    newpal = sns.color_palette("Set1", n_colors=3, desat=.5)
    sns.set_palette(list(reversed(newpal)))
    g = sns.PairGrid(colors, hue="sample", 
                     hue_kws={"marker": ["^", ".", "x"], 
                              "cmap":['Greens', 'Blues','Reds']})
    g.map_upper(plt.scatter, s=12, rasterized=True)
    g.map_diag(plt.hist, bins=30)
    #g.map_lower(sns.kdeplot, gridsize=60, n_levels=5, shade=False, shade_lowest=False);
    
    meanc = colors.mean()
    stdc = colors.std()
    for i,c in enumerate(train_cols):
        g.axes[0,i].set_xlim(meanc[c]-3*stdc[c], meanc[c]+3*stdc[c])
        g.axes[i,0].set_ylim(meanc[c]-3*stdc[c], meanc[c]+3*stdc[c])
            
    if filename is not None :
        root, ext = os.path.splitext(filename)
        filename2 = '{}_all{}'.format(root, ext)
        plt.savefig(filename2)
        plt.close()
    else :
        plt.show()
    
    return inTSCS

if __name__ == '__main__' :
    calc()