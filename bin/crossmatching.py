# -*- coding: utf-8 -*-

import os
import numpy as np
import subprocess as sp
from tqdm import tqdm

from astropy.table import Table, vstack, join, setdiff

import xmatch_http as xm


def xm_folders(xmatch_folder, nir_survey='2MASS') :
    folder_4cat = os.path.join(xmatch_folder, nir_survey.lower(), 
                               '3xmm_ps_wise_{}'.format(nir_survey.lower()))
    if not os.path.exists(folder_4cat) :
        os.makedirs(folder_4cat)

    folder_3catnir = os.path.join(xmatch_folder, nir_survey.lower(), 
                                  '3xmm_ps_{}'.format(nir_survey.lower()))
    if not os.path.exists(folder_3catnir) :
        os.makedirs(folder_3catnir)

    folder_3catmir = os.path.join(xmatch_folder, nir_survey.lower(), 
                                  '3xmm_ps_wise')
    if not os.path.exists(folder_3catmir) :
        os.makedirs(folder_3catmir)

    folder_2cat = os.path.join(xmatch_folder, nir_survey.lower(), '3xmm_ps')
    if not os.path.exists(folder_2cat) :
        os.makedirs(folder_2cat)

    return folder_2cat, folder_3catmir, folder_3catnir, folder_4cat

def merge_cats(folder_dict, nir_survey='2MASS', nir_label='NTM') :

    cat_folders = xm_folders(folder_dict['xmatch'], nir_survey)
    nirid = '{}objID'.format(nir_label)
    
    (xp_cat, xpw_cat, 
     xpn_cat, xpwn_cat) = (Table.read('{}.fits'.format(cat)) 
                           for cat in cat_folders)
    
    # Sources in XPW but not in XPWN    
    XPW_notXPWN = setdiff(xpw_cat, xpwn_cat, keys=['XMMSRCID', 'PSobjID', 'WSID'])
    XPW_notXPWN.rename_column('chi2Pos', 'chi2Pos_XPW')
    
    # Sources in XPN but not in XPWN
    XPN_notXPWN = setdiff(xpn_cat, xpwn_cat, keys=['XMMSRCID', 'PSobjID', nirid])
    XPN_notXPWN.rename_column('chi2Pos', 'chi2Pos_XPN')

    # Add 3-cat probabilities to common sources between XPWT and XPW
    xpw_cat_temp = xpw_cat[['chi2Pos', 'proba_XPW', 'XMMSRCID', 'PSobjID', 'WSID']]
    xpw_cat_temp.rename_column('chi2Pos', 'chi2Pos_XPW')
    xpwn_cat.remove_column('proba_XPW')
    xpwn_cat.rename_column('chi2Pos', 'chi2Pos_XPWN')
    XPWN_probaXPW = join(xpwn_cat, xpw_cat_temp, join_type='left',
                         keys=['XMMSRCID', 'PSobjID', 'WSID'])

    # Add 3-cat probabilities to common sources between XPWT and XPN
    xpn_cat_temp = xpn_cat[['chi2Pos', 'proba_XPN', 'XMMSRCID', 'PSobjID', nirid]]
    xpn_cat_temp.rename_column('chi2Pos', 'chi2Pos_XPN')
    XPWN_probaXPW.remove_column('proba_XPN')
    XPWN_probaXPW_probaXPN = join(XPWN_probaXPW, xpn_cat_temp, join_type='left',
                                  keys=['XMMSRCID', 'PSobjID', nirid])

    # Concat tables
    XPW_notXPWN.keep_columns(['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA', 
                              'chi2Pos_XPW', 'proba_XPW', 'nPos', 
                              'XMMSRCID', 'PSobjID', 'WSID'])

    XPN_notXPWN.keep_columns(['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA', 
                              'chi2Pos_XPN', 'proba_XPN', 'nPos', 
                              'XMMSRCID', 'PSobjID', nirid])

    XPWN_probaXPW_probaXPN.keep_columns(
                             ['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA', 
                              'chi2Pos_XPW', 'proba_XPW', 'chi2Pos_XPN', 
                              'proba_XPN', 'chi2Pos_XPWN', 'proba_XPWN', 
                              'nPos', 'XMMSRCID', 'PSobjID', 'WSID', nirid])

    XPWN_XPW_XPN = vstack([XPWN_probaXPW_probaXPN, XPW_notXPWN, XPN_notXPWN])

    # Sources in XP but not XPWN_XPW_XPN
    XP_notXPWN_XPW_XPN = setdiff(xp_cat, XPWN_XPW_XPN, 
                                 keys=['XMMSRCID', 'PSobjID'])
    XP_notXPWN_XPW_XPN.rename_column('chi2Pos', 'chi2Pos_XP')

    # Add 2-cat probabilities to common sources between XP and XPWN_XPW_XPN
    xp_cat_temp = xp_cat[['chi2Pos', 'proba_XP', 'XMMSRCID', 'PSobjID']]
    xp_cat_temp.rename_column('chi2Pos', 'chi2Pos_XP')
    XPWN_XPW_XPN_probaXP = join(XPWN_XPW_XPN, xp_cat_temp, join_type='left',
                                keys=['XMMSRCID', 'PSobjID'])
    
    # Concat tables
    XP_notXPWN_XPW_XPN.keep_columns(
                            ['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA', 
                             'chi2Pos_XP', 'proba_XP', 'nPos', 
                             'XMMSRCID', 'PSobjID'])

    merged_cat = vstack([XPWN_XPW_XPN_probaXP, XP_notXPWN_XPW_XPN])

    return merged_cat
    
def merge_bins(bin_stats, folder_dict, nir_survey='2MASS') :

    (folder_2cat, folder_3catmir, 
    folder_3catnir, folder_4cat) = xm_folders(folder_dict['xmatch'], nir_survey)

    xp_table = [None]*len(bin_stats)
    xpw_table = [None]*len(bin_stats)
    xpn_table = [None]*len(bin_stats)
    xpwn_table = [None]*len(bin_stats)
    
    for i, bin in enumerate(tqdm(bin_stats, desc='merging bins')) :
        bin_id = 'bin{}.fits'.format(str(int(bin['BIN_ID'])).zfill(3))

        xm_table = Table.read(os.path.join(folder_2cat, bin_id), memmap=True)
        xp_table[i] = xm_table

        xm_table = Table.read(os.path.join(folder_3catmir, bin_id), memmap=True)
        xpw_table[i] = xm_table[xm_table['nPos']==3]

        xm_table = Table.read(os.path.join(folder_3catnir, bin_id), memmap=True)
        xpn_table[i] = xm_table[xm_table['nPos']==3]

        xm_table = Table.read(os.path.join(folder_4cat, bin_id), memmap=True)
        xpwn_table[i] = xm_table[xm_table['nPos']==4]

    bins_cat = vstack(xp_table)
    bins_cat.write('{}.fits'.format(folder_2cat), overwrite=True)

    bins_cat = vstack(xpw_table)
    bins_cat.write('{}.fits'.format(folder_3catmir), overwrite=True)    

    bins_cat = vstack(xpn_table)
    bins_cat.write('{}.fits'.format(folder_3catnir), overwrite=True)
    
    bins_cat = vstack(xpwn_table)
    bins_cat.write('{}.fits'.format(folder_4cat), overwrite=True)        
    
def run(bin_stats, folder_dict, nir_survey='2MASS', nir_label='NTM') :
    """
    Run xmatch (web service) for all bins in bin_stats table.
    """
    
    (folder_2cat, folder_3catmir, 
    folder_3catnir, folder_4cat) = xm_folders(folder_dict['xmatch'], nir_survey)

    bins_folder = 'bins'
    if nir_survey is not '2MASS' :
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
        xmm['SC_POSERR'] = xmm['SC_POSERR']/np.sqrt(2)
        
        bin_file = 'temp_XMM.fits'
        xmm.write(bin_file, format='fits', overwrite=True)
        xm.addfile(bin_file)

        # Load Pan-STARRS bin and reject sources with high positional errors
        pstarrs_bin_file = os.path.join(folder_dict['pstarrs'], 
                                        bins_folder, bin_id)
        pstarrs = Table.read(pstarrs_bin_file)
        msk_poserr = np.logical_and(np.logical_and(pstarrs['e_RAJ2000'] < 5, 
                                                   pstarrs['e_RAJ2000'] > 0),
                                    np.logical_and(pstarrs['e_DEJ2000'] < 5, 
                                                   pstarrs['e_DEJ2000'] > 0))
        pstarrs = pstarrs[msk_poserr]

        bin_file = 'temp_PS.fits'
        pstarrs.write(bin_file, format='fits', overwrite=True)
        xm.addfile(bin_file)

        # Load WISE bin and reject sources with high positional errors
        wise_bin_file = os.path.join(folder_dict['wise'], bins_folder, bin_id)
        wise = Table.read(wise_bin_file)
        msk_poserr = np.logical_and(
                        np.logical_and(wise['eeMaj'] < 5, wise['eeMaj'] > 0),
                        np.logical_and(wise['eeMin'] < 5, wise['eeMin'] > 0))
        wise = wise[msk_poserr]
        
        bin_file = 'temp_WS.fits'
        wise.write(bin_file, format='fits', overwrite=True)
        xm.addfile(bin_file)

        # Load NIR bin and reject sources with high positional errors
        nir_bin_file = os.path.join(folder_dict['nir'], 'bins', bin_id)
        nir = Table.read(nir_bin_file)
        
        if nir_survey is '2MASS' :
            msk_poserr = np.logical_and(
                            np.logical_and(nir['errMaj'] < 5, nir['errMaj'] > 0),
                            np.logical_and(nir['errMin'] < 5, nir['errMin'] > 0))
        elif nir_survey is 'UKIDSS' :
            msk_poserr = np.logical_and(
                            np.logical_and(nir['sigRA'] < 5, nir['sigRA'] > 0),
                            np.logical_and(nir['sigDec'] < 5, nir['sigDec'] > 0))

        elif nir_survey is 'VISTA' :
            msk_poserr = [True]*len(nir)
        
        nir = nir[msk_poserr]
        
        bin_file = 'temp_{}.fits'.format(nir_label)
        nir.write(bin_file, format='fits', overwrite=True)
        xm.addfile(bin_file)

        ## Matching
        xm_table = xm.xmatch(xmm, pstarrs, catalog_prefixes=['XMM', 'PS'])
        xm_table.write(os.path.join(folder_2cat, bin_id), overwrite=True)
        
        xm_table = xm.xmatch(xmm, pstarrs, wise, 
                             catalog_prefixes=['XMM', 'PS', 'WS'])
        xm_table.write(os.path.join(folder_3catmir, bin_id), overwrite=True)
        
        xm_table = xm.xmatch(xmm, pstarrs, nir, 
                             catalog_prefixes=['XMM', 'PS', nir_label])
        xm_table.write(os.path.join(folder_3catnir, bin_id), overwrite=True)

        xm_table = xm.xmatch(xmm, pstarrs, wise, nir, 
                             catalog_prefixes=['XMM', 'PS', 'WS', nir_label])
        xm_table.write(os.path.join(folder_4cat, bin_id), overwrite=True)

        ## Remove files uploaded to the server and temp files
        sp.call('rm temp*', shell=True)
        xm.rmfile('temp_XMM.fits')        
        xm.rmfile('temp_PS.fits')
        xm.rmfile('temp_WS.fits')
        xm.rmfile('temp_{}.fits'.format(nir_label))

    xm.logout()
    


        