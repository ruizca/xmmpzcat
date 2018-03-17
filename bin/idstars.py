# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.table import Table, join, hstack
from hdbscan import HDBSCAN

from photoz import make_tpzinput, runtpz

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
sns.set(color_codes=True)

def plot_clusters(data, labels, colornames, filename=None):

    labelsize = 24
    ticksize = 18

    d_unc = data[labels < 0]
    d_star = data[labels == 0]
    d_glx = data[labels == 1]
    
    plt.figure()
    plt.scatter(d_unc[colornames[0]], d_unc[colornames[1]], color='DimGray', marker='*', lw=0)

    plt.scatter(d_star[colornames[0]], d_star[colornames[1]], color='YellowGreen', 
                edgecolors='none', marker='^', lw=0, rasterized=True)
    plt.scatter(d_star[colornames[0]], d_star[colornames[1]], color='DarkOliveGreen', 
                edgecolors='none', marker='^', lw=0, alpha=0.1, rasterized=True)
    
    plt.scatter(d_glx[colornames[0]], d_glx[colornames[1]], color='CornflowerBlue', 
                edgecolors='none', marker='o', s=5, lw=0, rasterized=True)
    plt.scatter(d_glx[colornames[0]], d_glx[colornames[1]], color='DarkSlateBlue', 
                edgecolors='none', marker='o', s=5, lw=0, alpha=0.1, rasterized=True)
    
    meanc = np.mean(data[colornames[0]])
    stdc = np.std(data[colornames[0]])
    plt.xlim(meanc - 4*stdc, meanc + 4*stdc)

    meanc = np.mean(data[colornames[1]])
    stdc = np.std(data[colornames[1]])
    plt.ylim(meanc - 4*stdc, meanc + 4*stdc)
    
    plt.xlabel('${}$'.format(colornames[0]), fontsize=labelsize)
    plt.ylabel('${}$'.format(colornames[1]), fontsize=labelsize)
        
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    
    plt.tight_layout()    
    
    if filename is None:
        #plt.show()
        pass
    else:
        plt.savefig(filename)
        plt.close()

def clusters(cat, mask, colors):

    table = cat[mask]
    table.keep_columns(colors)
    data = table.to_pandas()

    clusterer = HDBSCAN(min_cluster_size=20) #100 for real
    clusterer.fit(data)

    labels = Table.Column(clusterer.labels_, name='ct')
    proba = Table.Column(clusterer.probabilities_, name='prob_ct')
    stars = Table([cat[mask]['XMMSRCID'], labels, proba])
    
    return stars

    
def irsample(cat, sample='pstarrs', plotname=None):

    if sample == "pstarrs":
        mask_ptl = (cat["objInfoFlag"] & 0x00800000) == 0
    
    elif sample == "arches":
        mask_ptl = cat["ProbPSF"] == 1
    
    mask_opt = np.logical_and(cat['good_g'], cat['good_z'])
    mask_nir = np.logical_and(cat['good_J'], cat['good_K'])
    mask_mir = np.logical_and(cat['good_z'], cat['good_W1'])

    col_gmz = Table.Column(cat['gMag'] - cat['zMag'], name='g-z')
    col_zmw1 = Table.Column(cat['zMag'] - cat['W1Mag'], name='z-W1')
    col_jmk = Table.Column(cat['JMag'] - cat['KMag'], name='J-K')    
    cat.add_columns([col_gmz, col_zmw1, col_jmk])

    # Find MIR stars
    mask = np.logical_and(mask_ptl, np.logical_and(mask_opt, mask_mir))
    colors = ['g-z', 'z-W1']

    stars_mir = clusters(cat, mask, colors)

    plot_clusters(cat[mask], stars_mir['ct'], 
                  colornames=colors, filename=plotname) 
    
    # Find NIR stars
    mask = np.logical_and(mask_ptl, np.logical_and(mask_opt, mask_nir))
    colors = ['g-z', 'J-K']

    stars_nir = clusters(cat, mask, colors)

    plot_clusters(cat[mask], stars_nir['ct'], 
                  colornames=colors, filename=plotname) 
    

    stars = join(stars_mir, stars_nir, keys='XMMSRCID', 
                 join_type='outer', table_names=['mir', 'nir'])
    
    msk_stars = np.logical_or(np.logical_and(stars['ct_mir'] == 1,
                                             stars['prob_ct_mir'] > 0.5),
                              np.logical_and(stars['ct_nir'] == 1,
                                             stars['prob_ct_nir'] > 0.5))
    
    msk_nostars = np.logical_or(np.logical_and(stars['ct_mir'] == 0,
                                               stars['prob_ct_mir'] > 0.5),
                                np.logical_and(stars['ct_nir'] == 0,
                                               stars['prob_ct_nir'] > 0.5))
    
    stars_col = Table.Column(msk_stars, name='STARS_MIRNIR')
    nostars_col = Table.Column(msk_nostars, name='NOSTARS_MIRNIR')
    stars_table = Table([stars['XMMSRCID'], stars_col, nostars_col])

    idstar_sample = join(cat, stars_table, keys='XMMSRCID')
    mask = np.logical_or(idstar_sample['STARS_MIRNIR'], 
                         idstar_sample['NOSTARS_MIRNIR'])
    
    return idstar_sample[mask]

def make_training(cat, colors_names, train_file):
    
    err_names = ['e'+c for c in colors_names]
    colors = np.full((len(cat), len(colors_names)), np.nan)
    colors_err = np.full((len(cat), len(colors_names)), np.nan)    
    star_class = np.full((len(cat),), np.nan)
    
    for i, c in enumerate(colors_names):
        m1, m2 = c.split('m')
        colors[:, i] = cat[m1+'Mag'] - cat[m2+'Mag']
        colors_err[:, i] = np.sqrt(cat[m1+'MagErr']**2 + cat[m2+'MagErr']**2)
    
    star_class[cat['STARS_MIRNIR']] = np.ones(len(np.nonzero(cat['STARS_MIRNIR'])))
    star_class[cat['NOSTARS_MIRNIR']] = np.zeros(len(np.nonzero(cat['NOSTARS_MIRNIR'])))
    
    sample = hstack([Table([cat['PSobjID']]), Table(colors), 
                     Table(colors_err), Table([star_class])])
    
    colnames = ['id'] + colors_names + err_names + ['star']
    sample = Table(sample, names=colnames)
    
    sample.write(train_file, format='ascii.commented_header', overwrite=True)
    
    return colnames

def make_testing(cat, columns, test_file):

    colors_names = [c for c in columns[1:-1] if not c.startswith('e')]
    colors = np.full((len(cat), len(colors_names)), np.nan)
    colors_err = np.full((len(cat), len(colors_names)), np.nan)

    for i, c in enumerate(colors_names):
        m1, m2 = c.split('m')
        colors[:, i] = cat[m1+'Mag'] - cat[m2+'Mag']
        colors_err[:, i] = np.sqrt(cat[m1+'MagErr']**2 + cat[m2+'MagErr']**2)

    msk = np.logical_and(cat['good_opt'], (cat['objInfoFlag'] & 0x00800000) == 0)
        
    sample = hstack([Table([cat['PSobjID']]), Table(colors), Table(colors_err)])
    sample = Table(sample, names=columns[:-1])
    
    sample[msk].write(test_file, format='ascii.commented_header', overwrite=True)
    
    return Table([sample[msk][columns[0]]])
    
def find(cat):

    colors = ['gmr', 'gmi', 'gmz', 'gmy', 'rmi', 
              'rmz', 'rmy', 'imz', 'imy', 'zmy']
    
    train_file = '../photoz/samples/training/stars.dat'
    test_file = '../photoz/samples/stars.dat'
    run_file = '../photoz/runs/stars.input'
    
    results_folder = '../photoz/results/stars'
    results = os.path.join(results_folder, 'stars')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    training = irsample(cat)
    columns = make_training(training, colors, train_file)    
    testids = make_testing(cat, columns, test_file)
    make_tpzinput(run_file, train_file, test_file, results, columns=columns, 
                  zmin=0, zmax=2, znbins=2, pmode='TPZ_C', pclass='Class')
    runtpz(run_file, testids, results, pdfs=False)
    
    stars_tpz = Table.read(results + '.fits', memmap=True)
    msk = stars_tpz['zmode0'] == 1
    stars_tpz = Table([stars_tpz[msk]['id'], msk[msk]], 
                      names=['PSobjID', 'STARS_TPZ'])
    
    stars_nirmir = training[training['STARS_MIRNIR']]
    stars_nirmir.keep_columns(['PSobjID', 'STARS_MIRNIR'])

    print(stars_tpz)
    print(stars_nirmir)
    
    stars = join(stars_nirmir, stars_tpz, keys='PSobjID', join_type='outer')
    print(stars)
    