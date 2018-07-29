# -*- coding: utf-8 -*-
"""
Functions for the cross-matching of 
catalogues using the xmatch web server.
"""
import os
import subprocess as sp

from tqdm import tqdm
from astropy.table import Table, vstack, join, setdiff
import numpy as np

import xmatch_http as xm


def xm_folders(xmatch_folder, opt_survey='pstarrs', nir_survey='2MASS'):
    """
    Define and create directory structure for the cross-matching results.
    """
    folder_dict = {}
    nir_survey = nir_survey.lower()

    dirname = '3xmm_{}_wise_{}'.format(opt_survey, nir_survey)
    folder_dict['4cat'] = os.path.join(xmatch_folder, nir_survey, dirname)
    if not os.path.exists(folder_dict['4cat']):
        os.makedirs(folder_dict['4cat'])

    dirname = '3xmm_{}_{}'.format(opt_survey, nir_survey)
    folder_dict['3catnir'] = os.path.join(xmatch_folder, nir_survey, dirname)
    if not os.path.exists(folder_dict['3catnir']):
        os.makedirs(folder_dict['3catnir'])


    dirname = '3xmm_{}_wise'.format(opt_survey)
    folder_dict['3catmir'] = os.path.join(xmatch_folder, nir_survey, dirname)
    if not os.path.exists(folder_dict['3catmir']):
        os.makedirs(folder_dict['3catmir'])

    dirname = '3xmm_{}'.format(opt_survey)
    folder_dict['2cat'] = os.path.join(xmatch_folder, nir_survey, dirname)
    if not os.path.exists(folder_dict['2cat']):
        os.makedirs(folder_dict['2cat'])

    return folder_dict


def merge_cats(folder_dict, opt_survey='pstarrs', opt_label='PS',
               nir_survey='2MASS', nir_label='NTM'):

    catdir = xm_folders(folder_dict['xmatch'], nir_survey)
    optid = '{}objID'.format(opt_label)
    nirid = '{}objID'.format(nir_label)

    xo_cat = Table.read('{}.fits'.format(catdir['2cat']))
    xow_cat = Table.read('{}.fits'.format(catdir['3catmir']))
    xon_cat = Table.read('{}.fits'.format(catdir['3catnir']))
    xown_cat = Table.read('{}.fits'.format(catdir['4cat']))

    # Sources in XOW but not in XOWN
    XOW_notXOWN = setdiff(xow_cat, xown_cat, keys=['XMMSRCID', optid, 'WSID'])
    XOW_notXOWN.rename_column('chi2Pos', 'chi2Pos_XOW')

    # Sources in XON but not in XOWN
    XON_notXOWN = setdiff(xon_cat, xown_cat, keys=['XMMSRCID', optid, nirid])
    XON_notXOWN.rename_column('chi2Pos', 'chi2Pos_XON')

    # Add 3-cat probabilities to common sources between XOWT and XOW
    xow_cat_temp = xow_cat[['chi2Pos', 'proba_XOW', 'XMMSRCID', optid, 'WSID']]
    xow_cat_temp.rename_column('chi2Pos', 'chi2Pos_XOW')
    xown_cat.remove_column('proba_XOW')
    xown_cat.rename_column('chi2Pos', 'chi2Pos_XOWN')
    XOWN_probaXOW = join(xown_cat, xow_cat_temp, join_type='left',
                         keys=['XMMSRCID', optid, 'WSID'])

    # Add 3-cat probabilities to common sources between XOWT and XON
    xon_cat_temp = xon_cat[['chi2Pos', 'proba_XON', 'XMMSRCID', optid, nirid]]
    xon_cat_temp.rename_column('chi2Pos', 'chi2Pos_XON')
    XOWN_probaXOW.remove_column('proba_XPN')
    XOWN_probaXOW_probaXON = join(XOWN_probaXOW, xon_cat_temp, join_type='left',
                                  keys=['XMMSRCID', optid, nirid])

    # Concat tables
    XOW_notXOWN.keep_columns(['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA',
                              'chi2Pos_XOW', 'proba_XOW', 'nPos',
                              'XMMSRCID', optid, 'WSID'])

    XON_notXOWN.keep_columns(['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA',
                              'chi2Pos_XON', 'proba_XON', 'nPos',
                              'XMMSRCID', optid, nirid])

    XOWN_probaXOW_probaXON.keep_columns(
                             ['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA',
                              'chi2Pos_XOW', 'proba_XOW', 'chi2Pos_XON',
                              'proba_XON', 'chi2Pos_XOWN', 'proba_XOWN',
                              'nPos', 'XMMSRCID', optid, 'WSID', nirid])

    XOWN_XOW_XON = vstack([XOWN_probaXOW_probaXON, XOW_notXOWN, XON_notXOWN])

    # Sources in XO but not XOWN_XOW_XON
    XO_notXOWN_XOW_XON = setdiff(xo_cat, XOWN_XOW_XON, 
                                 keys=['XMMSRCID', optid])
    XO_notXOWN_XOW_XON.rename_column('chi2Pos', 'chi2Pos_XO')

    # Add 2-cat probabilities to common sources between XO and XOWN_XOW_XON
    xo_cat_temp = xo_cat[['chi2Pos', 'proba_XO', 'XMMSRCID', optid]]
    xo_cat_temp.rename_column('chi2Pos', 'chi2Pos_XO')
    XOWN_XOW_XON_probaXO = join(XOWN_XOW_XON, xo_cat_temp, join_type='left',
                                keys=['XMMSRCID', optid])
    
    # Concat tables
    XO_notXOWN_XOW_XON.keep_columns(
                            ['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA',
                             'chi2Pos_XO', 'proba_XO', 'nPos', 
                             'XMMSRCID', optid])

    merged_cat = vstack([XOWN_XOW_XON_probaXO, XO_notXOWN_XOW_XON])

    return merged_cat


def merge_bins(bin_stats, folder_dict, 
               opt_survey='pstarrs', nir_survey='2MASS'):
    """
    Combine the results of the cross-match for each bin in a single table.
    """
    catdir = xm_folders(folder_dict['xmatch'], opt_survey, nir_survey)
    xo_table = [None]*len(bin_stats)
    xow_table = [None]*len(bin_stats)
    xon_table = [None]*len(bin_stats)
    xown_table = [None]*len(bin_stats)
    
    for i, sbin in enumerate(tqdm(bin_stats, desc='merging bins')):
        bin_id = 'bin{}.fits'.format(str(int(sbin['BIN_ID'])).zfill(3))

        xm_table = Table.read(os.path.join(catdir['2cat'], bin_id), memmap=True)
        xo_table[i] = xm_table

        xm_table = Table.read(os.path.join(catdir['3catmir'], bin_id), memmap=True)
        xow_table[i] = xm_table[xm_table['nPos']==3]

        xm_table = Table.read(os.path.join(catdir['3catnir'], bin_id), memmap=True)
        xon_table[i] = xm_table[xm_table['nPos']==3]

        xm_table = Table.read(os.path.join(catdir['4cat'], bin_id), memmap=True)
        xown_table[i] = xm_table[xm_table['nPos']==4]

    bins_cat = vstack(xo_table)
    bins_cat.write('{}.fits'.format(catdir['2cat']), overwrite=True)

    bins_cat = vstack(xow_table)
    bins_cat.write('{}.fits'.format(catdir['3catmir']), overwrite=True)    

    bins_cat = vstack(xon_table)
    bins_cat.write('{}.fits'.format(catdir['3catnir']), overwrite=True)

    bins_cat = vstack(xown_table)
    bins_cat.write('{}.fits'.format(catdir['4cat']), overwrite=True)        


def run(bin_stats, folder_dict,
        opt_survey='pstarrs', opt_label='PS',
        nir_survey='2MASS', nir_label='NTM'):
    """
    Run xmatch (web service) for all bins in bin_stats table.
    """
    catdir = xm_folders(folder_dict['xmatch'], opt_survey, nir_survey)
    bins_folder = 'bins'
    if nir_survey != '2MASS':
        bins_folder += '_{}'.format(nir_survey.lower())

    xm.login()
    for i, bin in enumerate(tqdm(bin_stats, desc='xmatching')) :
        bin_id = 'bin{}.fits'.format(str(int(bin['BIN_ID'])).zfill(3))

        ## Load bins
        # Load XMM bin and reject sources with high positional errors
        xmm_bin_file = os.path.join(folder_dict['xmm'], bins_folder, bin_id)
        xmm = Table.read(xmm_bin_file)
        msk_poserr = np.logical_and(xmm['SC_POSERR'] > 0, xmm['SC_POSERR'] < 10)
        xmm = xmm[msk_poserr]

        # Correct X-ray positional error (Pineau+2017)
        xmm['SC_POSERR'] = xmm['SC_POSERR']/np.sqrt(2)

        bin_file = 'temp_XMM.fits'
        xmm.write(bin_file, format='fits', overwrite=True)
        xm.addfile(bin_file)


        # Load optical bin and reject sources with high positional errors
        opt_bin_file = os.path.join(folder_dict['opt'], bins_folder, bin_id)
        opt = Table.read(opt_bin_file)
        
        if opt_survey == 'pstarrs':
            msk_poserr = np.logical_and(np.logical_and(opt['e_RAJ2000'] < 5, 
                                                       opt['e_RAJ2000'] > 0),
                                        np.logical_and(opt['e_DEJ2000'] < 5, 
                                                       opt['e_DEJ2000'] > 0))
        elif opt_survey == 'sdss':
            msk_poserr = np.logical_and(np.logical_and(opt['raErr'] < 5, 
                                                       opt['raErr'] > 0),
                                        np.logical_and(opt['decErr'] < 5, 
                                                       opt['decErr'] > 0))
            # Add systematics to positional errors
            opt['raErr'] = np.sqrt(opt['raErr']**2 + 0.01)
            opt['decErr'] = np.sqrt(opt['decErr']**2 + 0.01)

        else:
            raise ValueError('Unknown optical survey!')

        opt = opt[msk_poserr]

        bin_file = 'temp_{}.fits'.format(opt_label)
        opt.write(bin_file, format='fits', overwrite=True)
        xm.addfile(bin_file)

        # Load WISE bin and reject sources with high positional errors
        wise_bin_file = os.path.join(folder_dict['wise'], bins_folder, bin_id)
        wise = Table.read(wise_bin_file)
        msk_poserr = np.logical_and(np.logical_and(wise['eeMaj'] < 5,
                                                   wise['eeMaj'] > 0),
                                    np.logical_and(wise['eeMin'] < 5,
                                                   wise['eeMin'] > 0))
        ## Include systematic error
        wise['eeMaj'] = np.sqrt(wise['eeMaj']**2 + 0.01)
        wise['eeMin'] = np.sqrt(wise['eeMin']**2 + 0.01)

        wise = wise[msk_poserr]

        bin_file = 'temp_WS.fits'
        wise.write(bin_file, format='fits', overwrite=True)
        xm.addfile(bin_file)


        # Load NIR bin and reject sources with high positional errors
        nir_bin_file = os.path.join(folder_dict['nir'], 'bins', bin_id)
        nir = Table.read(nir_bin_file)
        
        if nir_survey == '2MASS':
            msk_poserr = np.logical_and(
                            np.logical_and(nir['errMaj'] < 5, nir['errMaj'] > 0),
                            np.logical_and(nir['errMin'] < 5, nir['errMin'] > 0))

        elif nir_survey == 'UKIDSS':
            msk_poserr = np.logical_and(
                            np.logical_and(nir['sigRA'] < 5, nir['sigRA'] > 0),
                            np.logical_and(nir['sigDec'] < 5, nir['sigDec'] > 0))

        elif nir_survey == 'VISTA':
            msk_poserr = [True]*len(nir)

        else:
            raise ValueError('Unknown NIR survey!')
        
        nir = nir[msk_poserr]
        
        bin_file = 'temp_{}.fits'.format(nir_label)
        nir.write(bin_file, format='fits', overwrite=True)
        xm.addfile(bin_file)


        ## Matching
        xm_table = xm.xmatch(xmm, opt, catalog_prefixes=['XMM', opt_label])
        xm_table.write(os.path.join(catdir['2cat'], bin_id),
                       format='fits', overwrite=True)

        xm_table = xm.xmatch(xmm, opt, wise,
                             catalog_prefixes=['XMM', opt_label, 'WS'])
        xm_table.write(os.path.join(catdir['3catmir'], bin_id),
                       format='fits', overwrite=True)

        xm_table = xm.xmatch(xmm, opt, nir,
                             catalog_prefixes=['XMM', opt_label, nir_label])
        xm_table.write(os.path.join(catdir['3catnir'], bin_id),
                       format='fits', overwrite=True)

        xm_table = xm.xmatch(xmm, opt, wise, nir,
                             catalog_prefixes=['XMM', opt_label, 'WS', nir_label])
        xm_table.write(os.path.join(catdir['4cat'], bin_id),
                       format='fits', overwrite=True)


        ## Remove files uploaded to the server and temp files
        sp.call('rm temp*', shell=True)
        xm.rmfile('temp_XMM.fits')
        xm.rmfile('temp_{}.fits'.format(opt_label))
        xm.rmfile('temp_WS.fits')
        xm.rmfile('temp_{}.fits'.format(nir_label))

    xm.logout()
