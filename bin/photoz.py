# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 20:01:17 2018

@author: alnoah
"""

import os
import subprocess as sp
from itertools import product

import numpy as np
from tqdm import tqdm
from astropy.table import Table, hstack, vstack

import colorcovering as cc

def make_training(cat_train, label, train_file, exttype):
    
    opt_colors = ['gmr', 'gmi', 'gmz', 'rmi', 'rmz', 'imz', 'imy'] 
    nir_colors = ['zmj', 'jmh']
    mir_colors = ['zmw1', 'w1mw2']
    nirmir_colors = ['zmw1', 'w1mw2', 'kmw1', 'jmw1', 'hmw1', 'jmh', 'hmk', 'jmk']

    msk_opt1 = np.logical_and(cat_train['good_gmr'], cat_train['good_gmi'])
    msk_opt2 = np.logical_and(cat_train['good_gmz'], cat_train['good_rmi'])
    msk_opt3 = np.logical_and(cat_train['good_rmz'], cat_train['good_imz'])
    msk_opt = np.logical_and(np.logical_and(msk_opt1, msk_opt2),
                             np.logical_and(msk_opt3, cat_train['good_imy']))

    if label == 'xp':
        colors = opt_colors
        color_names = opt_colors
        msk = msk_opt
        
    elif label == 'xpw':
        colors = opt_colors + mir_colors
        color_names = opt_colors + ['zmW1', 'W1mW2']
        msk = np.logical_and(np.logical_and(cat_train['good_zmw1'], 
                                            cat_train['good_w1mw2']),
                             msk_opt)
    elif label == 'xpn':
        colors = opt_colors + nir_colors
        color_names = opt_colors + ['zmJ', 'JmH']
        msk = np.logical_and(np.logical_and(cat_train['good_zmj'], 
                                            cat_train['good_jmh']),
                             msk_opt)
    elif label == 'xpwn':
        colors = opt_colors + nirmir_colors
        color_names = opt_colors + ['zmW1', 'W1mW2', 'KmW1', 'JmW1', 
                                    'HmW1', 'JmH', 'HmK', 'JmK']
        msk1 = np.logical_and(cat_train['good_zmw1'], cat_train['good_w1mw2'])
        msk2 = np.logical_and(cat_train['good_kmw1'], cat_train['good_jmw1'])
        msk3 = np.logical_and(cat_train['good_hmw1'], cat_train['good_jmh'])
        msk4 = np.logical_and(cat_train['good_hmk'], cat_train['good_jmk'])
        msk5 = np.logical_and(np.logical_and(msk1, msk2),
                              np.logical_and(msk3, msk4))
        msk = np.logical_and(msk_opt, msk5)
        
    else:
        raise ValueError('Unknown label!')

    if exttype == 'ext':
        msk = np.logical_and(msk, (cat_train['objInfoFlag'] & 0x00800000) != 0)
    elif exttype == 'ptl':
        msk = np.logical_and(msk, (cat_train['objInfoFlag'] & 0x00800000) == 0)
    else:
        raise ValueError('Unknown exttype!')
        
    
    errors = ['{}Err'.format(x) for x in colors]
    error_names = ['e{}'.format(x) for x in color_names]
    columns = ['PSTARRS_objID'] + colors + errors + ['zsp']
    column_names = ['id'] + color_names + error_names + ['zsp']
    
    sample = Table(cat_train[columns], names=column_names)
    sample[msk].write(train_file, format='ascii.commented_header', overwrite=True)
    
    return column_names

def make_testing(cat, columns, label, test_file, exttype, dered=True):

    colors_names = [c for c in columns[1:-1] if not c.startswith('e')]
    colors = np.full((len(cat), len(colors_names)), np.nan)
    colors_err = np.full((len(cat), len(colors_names)), np.nan)
    reddening = np.full((len(cat), len(colors_names)), np.nan)

    for i, c in enumerate(colors_names):
        m1, m2 = c.split('m')
        colors[:, i] = cat[m1+'Mag'] - cat[m2+'Mag']
        colors_err[:, i] = np.sqrt(cat[m1+'MagErr']**2 + cat[m2+'MagErr']**2)
        reddening[:, i] = cat['A'+m1] - cat['A'+m2]

    if dered:
        colors = colors - reddening

    if label == 'xp':
        msk = cat['sample_5mags']
        
    elif label == 'xpw':
        msk = cat['sample_7mags']
        
    elif label == 'xpn':
        msk = cat['sample_8mags']
        
    elif label == 'xpwn':
        msk = cat['sample_10mags']
        
    else:
        raise ValueError('Unknown label!')

    if exttype == 'ext':
        msk = np.logical_and(msk, (cat['objInfoFlag'] & 0x00800000) != 0)
    elif exttype == 'ptl':
        msk = np.logical_and(msk, (cat['objInfoFlag'] & 0x00800000) == 0)
    else:
        raise ValueError('Unknown exttype!')
        
    sample = hstack([Table([cat['PSobjID']]), Table(colors), Table(colors_err)])
    sample = Table(sample, names=columns[:-1])
    
    sample[msk].write(test_file, format='ascii.commented_header', overwrite=True)
    
    return Table([sample[msk][columns[0]]])

def make_tpzinput(file, trainfile, testfile, finalfile, columns, errors=True,
                  zmin=0.001, zmax=1.0, znbins=25,
                  nrandom=6, ntress=8, natt=4,
                  rms=0.06, sigma=3.0,
                  pmode='TPZ', pclass='Reg'):
    
    keyatt = columns[-1]
    columns_test = ','.join(columns[:-1])
    if errors:
        att = ','.join([c for c in columns[1:-1] if not c.startswith('e')])
    else:
        att = ','.join([c for c in columns[1:-1]])
        
    columns = ','.join(columns)
    
    with open(file, 'w+') as fp:
        fp.write('TrainFile        : {}\n'.format(os.path.basename(trainfile)))
        fp.write('TestFile         : {}\n'.format(os.path.basename(testfile)))
        fp.write('FinalFileName    : {}\n'.format(os.path.basename(finalfile)))
        fp.write('Path_Train       : {}\n'.format(os.path.dirname(trainfile)))
        fp.write('Path_Test        : {}\n'.format(os.path.dirname(testfile)))
        fp.write('Path_Output      : {}\n'.format(os.path.dirname(finalfile)))
        fp.write('Columns          : {}\n'.format(columns))
        fp.write('Att              : {}\n'.format(att))
        fp.write('Columns_Test     : {}\n'.format(columns_test))
        fp.write('KeyAtt           : {}\n'.format(keyatt))
        fp.write('CheckOnly        : no\n')
        fp.write('PredictionMode   : {}\n'.format(pmode))
        fp.write('PredictionClass  : {}\n'.format(pclass))
        fp.write('MinZ             : {:f}\n'.format(zmin))
        fp.write('MaxZ             : {:f}\n'.format(zmax))
        fp.write('NzBins           : {:f}\n'.format(znbins))
        fp.write('NRandom          : {:d}\n'.format(nrandom))
        fp.write('NTrees           : {:d}\n'.format(ntress))
        fp.write('Natt             : {:d}\n'.format(natt))
        fp.write('OobError         : yes\n')
        fp.write('VarImportance    : yes\n')
        fp.write('MinLeaf          : 5\n')
        fp.write('ImpurityIndex    : entropy\n')
        fp.write('Topology         : hex\n')
        fp.write('Periodic         : yes\n')
        fp.write('Ntop             : 15\n')
        fp.write('Iterations       : 200\n')
        fp.write('SomType          : online\n')
        fp.write('AlphaStart       : 0.9\n')
        fp.write('AlphaEnd         : 0.5\n')
        fp.write('ImportanceFile   : none\n')
        fp.write('SigmaFactor      : {:f}\n'.format(sigma))
        fp.write('RmsFactor        : {:f}\n'.format(rms))
        fp.write('WriteFits        : yes\n')
        fp.write('MultipleFiles    : no\n')
        fp.write('SparseRep        : no\n')
        fp.write('SparseDims       : 200,50,2\n')
        fp.write('NumberCoef       : 32001\n')
        fp.write('NumberBases      : 20\n')
        fp.write('OriginalPdfFile  : yes\n')

def runtpz(inputfile, srcids, resultfile, pdfs=True):
    
    sp.call('mpirun -n 2 runMLZ {}'.format(inputfile), shell=True)

    folder = os.path.dirname(resultfile)
    file = os.path.basename(resultfile)

    # Add ids to photoz table    
    photoz_file = os.path.join(folder, 'results', file + '.0.mlz')
    data = Table.read(photoz_file, format='ascii')
    data = hstack([srcids, data])
    data.write(os.path.join(folder, '{}.fits'.format(file)),
               format='fits', overwrite=True)

    if pdfs:
        # Add ids to pdfs table, and add column with the corresponding z values
        pdf_file = os.path.join(folder, 'results', file + '.0.P.fits')
        data = Table.read(pdf_file, format='fits')
        z_vals = np.tile(data['PDF values'][-1], (len(srcids), 1))
        data = hstack([srcids, Table([z_vals]), Table([data['PDF values'][:-1]])])
        data.write(os.path.join(folder, '{}_pdfs.fits'.format(file)),
                   format='fits', overwrite=True)

def pdf_params(pdfs):    
    Npeaks = np.ones((len(pdfs),))
    peakStrength = np.ones((len(pdfs),))
    zphot_peak2 = np.full((len(pdfs),), np.nan)

    for i, pdf in enumerate(tqdm(pdfs, desc='Calculating PDF parameters')):
        ext_pdf = np.concatenate([np.zeros((5)), pdf[2], np.zeros((5))])
        peaks = (np.diff(np.sign(np.diff(ext_pdf))) < 0).nonzero()[0] + 1
        Npeaks[i] = len(peaks)

        if Npeaks[i] > 1:
            pdf_peaks = ext_pdf[peaks]
            idx_max = np.argmax(pdf_peaks)
            Pmax = pdf_peaks[idx_max]
            	
            peaks_nomax = np.delete(peaks, idx_max)
            pdf_peaks_nomax = ext_pdf[peaks_nomax]
            idx_max2 = np.argmax(pdf_peaks_nomax)
            P2 = pdf_peaks_nomax[idx_max2]
            
            peakStrength[i] = 1-P2/Pmax
            zphot_peak2[i] = pdf[1][peaks_nomax[idx_max2]-5]
                
    params = Table([peakStrength, Npeaks, zphot_peak2], 
                   names=['PHOT_PS', 'PHOT_NP', 'PHOT_Z2'])
    
    return params

def calc(catalogue, photoz_folder):
        
    samples_folder = os.path.join(photoz_folder, 'samples')
    if not os.path.exists(samples_folder):
        os.makedirs(samples_folder)

    training_folder = os.path.join(samples_folder, 'training')
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
        
    runs_folder = os.path.join(photoz_folder, 'runs')
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    results_folder = os.path.join(photoz_folder, 'results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    training_file = 'xtraining_PSTARRS1_SDSS_VISTA_UKIDSS_2MASS_WISE_coloursDered.fits'
    training_file = os.path.join(photoz_folder, training_file)    
    training_cat = Table.read(training_file, memmap=True)
    
    samples = ['xp', 'xpw', 'xpn', 'xpwn']
    photoz_tables = [None]*len(samples)*2
    
    i = 0
    for s, e in product(samples, ['ext', 'ptl']):
        
        train = os.path.join(training_folder, '{}_{}.dat'.format(s, e))
        columns = make_training(training_cat, s, train, e)
        
        test = os.path.join(samples_folder, '{}_{}.dat'.format(s, e))
        testids = make_testing(catalogue, columns, s, test, e)

        results = os.path.join(results_folder, '{}_{}'.format(s, e))
        if not os.path.exists(results):
            os.makedirs(results)
        results = os.path.join(results, 'photoz')

        colcol_filename = '../plots/colcol_{}_{}.png'.format(s, e)
        inTSCS = cc.calc(train, test, colcol_filename)

        if e == 'ext':
            zmax = 1.0
            znbins = 25
        else:
            zmax = 3.0
            znbins = 75

        tpzinput = os.path.join(runs_folder, '{}_{}.input'.format(s, e))
        make_tpzinput(tpzinput, train, test, results, columns=columns, 
                      zmax=zmax, znbins=znbins, nrandom=2, ntress=2)
    
        runtpz(tpzinput, testids, results)
     
        photoz_tables[i] = Table.read(results + '.fits', memmap=True)
        pdfs = Table.read(results + '_pdfs.fits', memmap=True)
        
        pdfparams = pdf_params(pdfs)
        photoz_tables[i] = hstack([photoz_tables[i], pdfparams, inTSCS])

        i += 1

    photoz = vstack(photoz_tables)
    
    return photoz    
    