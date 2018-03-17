# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:49:20 2018

@author: alnoah
"""

import os
import numpy as np

import matplotlib
matplotlib.use("qt5agg")

from astropy.table import Table, vstack, unique
from astropy.coordinates import SkyCoord, ICRS
from astropy import units as u
from astropy_healpix import HEALPix

from pymoc import MOC
from pymoc.io.fits import read_moc_fits

import utils
import xmmmocs
import getdata
import binning
import xposcorr
import crossmatching


def clean_cat(catalogue, probalimit=0.68):
    
    catcoords = SkyCoord(ra=catalogue['posRA'], 
                         dec=catalogue['posDec'])

    exgal_flag = np.abs(catcoords.galactic.b.value) > 20
    proba_xp = catalogue['proba_XP']
    proba_xpw = catalogue['proba_XPW']
    proba_xpn = catalogue['proba_XPN']
    proba_xpwn = catalogue['proba_XPWN']

    # Define samples by number of counterparts    
    sampleXPWN = np.logical_and(exgal_flag, proba_xpwn > probalimit)
    
    sampleXPW = np.logical_and(exgal_flag, ~sampleXPWN)
    sampleXPW = np.logical_and(sampleXPW, proba_xpw > probalimit)
    sampleXPW = np.logical_and(sampleXPW, np.logical_or(proba_xpn <= proba_xpw,
                                                        np.isnan(proba_xpn)))
    
    sampleXPN = np.logical_and(~sampleXPWN, ~sampleXPW)
    sampleXPN = np.logical_and(sampleXPN, exgal_flag)
    sampleXPN = np.logical_and(sampleXPN, proba_xpn > probalimit)
    sampleXPN = np.logical_and(sampleXPN, np.logical_or(proba_xpw <= proba_xpn,
                                                        np.isnan(proba_xpw)))
    
    sampleXP = np.logical_and(~sampleXPWN, ~sampleXPW)
    sampleXP = np.logical_and(sampleXP, ~sampleXPN)
    sampleXP = np.logical_and(sampleXP, exgal_flag)
    sampleXP = np.logical_and(sampleXP, proba_xp > probalimit)

    samplePZ = np.logical_or(sampleXPWN, sampleXPW)
    samplePZ = np.logical_or(samplePZ, sampleXPN)
    samplePZ = np.logical_or(samplePZ, sampleXP)
    
    catalogue.add_column(Table.Column(exgal_flag), name='exgal')
    catalogue.add_column(Table.Column(sampleXP), name='sampleXP')
    catalogue.add_column(Table.Column(sampleXPW), name='sampleXPW')
    catalogue.add_column(Table.Column(sampleXPN), name='sampleXPN')
    catalogue.add_column(Table.Column(sampleXPWN), name='sampleXPWN')
    catalogue.add_column(Table.Column(samplePZ), name='sample_photoz')

    # Order table by decreasing probabilities
    # It uses columns with negative probabilities, 
    # since 'sort' works in ascending order
    catalogue.add_column(-catalogue['proba_XPWN'], name='mproba_XPWN')
    catalogue.add_column(-catalogue['proba_XPW'], name='mproba_XPW')
    catalogue.add_column(-catalogue['proba_XPN'], name='mproba_XPN')
    catalogue.add_column(-catalogue['proba_XP'], name='mproba_XP')
    
    catalogue.sort(['mproba_XPWN','mproba_XPW', 'mproba_XPN', 'mproba_XP'])
    catalogue.remove_columns(['mproba_XPWN','mproba_XPW', 
                              'mproba_XPN', 'mproba_XP'])

    # Remove duplicate sources
    catalogue = unique(catalogue, keys='XMMSRCID')
    catalogue = unique(catalogue, keys='PSobjID')
    
    # Split table between sources with and without WISE counterparts and filter
    # (unique doesn't work with masked columns)
    good_w = ~catalogue['WSID'].mask    
    catalogue_badw = catalogue[~good_w]    
    catalogue_goodw = catalogue[good_w]
    catalogue_goodw = unique(catalogue_goodw, keys='WSID')
    catalogue = vstack([catalogue_goodw, catalogue_badw])

    # The same for NIR counterparts
    good_n = ~catalogue['NIRobjID'].mask    
    catalogue_badn = catalogue[~good_n]    
    catalogue_goodn = catalogue[good_n]
    catalogue_goodn = unique(catalogue_goodn, keys='NIRobjID')
    catalogue = vstack([catalogue_goodn, catalogue_badn])

    return catalogue

def merge_cat(tmass, ukidss, vista):
    moc_vista = MOC()
    read_moc_fits(moc_vista, '../data/vista/moc_vista.fits')
    moc_ukidss = MOC()
    read_moc_fits(moc_ukidss, '../data/ukidss/moc_ukidss.fits')

    moc_order = moc_vista.order
    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())
    moc_allsky = MOC(0, tuple(range(12)))
    moc_tmass = moc_allsky - moc_ukidss - moc_vista
    moc_vista = moc_vista - moc_ukidss    
    
    vista_final = utils.sources_inmoc(vista, hp, moc_vista, 
                                      moc_order=moc_order, 
                                      ra='posRA', dec='posDec')
    vista_final.rename_column('NVTobjID', 'NIRobjID')    
    vista_final.add_column(Table.Column(['VISTA']*len(vista_final), 
                                        name='NIR_SURVEY'))

    tmass_final = utils.sources_inmoc(tmass, hp, moc_tmass, 
                                      moc_order=moc_order, 
                                      ra='posRA', dec='posDec')
    tmass_final.rename_column('NTMobjID', 'NIRobjID')    
    tmass_final.add_column(Table.Column(['2MASS']*len(tmass_final), 
                                        name='NIR_SURVEY'))

    ukidss.rename_column('NUKobjID', 'NIRobjID')    
    ukidss.add_column(Table.Column(['UKIDSS']*len(ukidss), name='NIR_SURVEY'))

    xmatchcat_final = vstack([tmass_final, vista_final, ukidss])

    msk = np.logical_or(xmatchcat_final['NIR_SURVEY'] == 'UKIDSS',
                        xmatchcat_final['NIR_SURVEY'] == '2MASS')
    print(len(np.where(msk)))
    msk = np.logical_and(xmatchcat_final['NIRobjID'].mask, msk)
    print(len(np.where(msk)))

    return xmatchcat_final

def make_cat(nir_survey, radius=15*u.arcmin, poscorr=False, make_mocs=False, 
             getXdata=False, getPSdata=False, getWSdata=False, getNIRdata=False,
             define_bins=False, make_bins=False, make_xmatch=False):
        
    cat_url = 'http://xmmssc.irap.omp.eu/Catalogue/3XMM-DR7'
    obs_filename = '3xmmdr7_obslistV2.fits'
    det_filename = '3XMM_DR7cat_v1.0_test.fits'
    src_filename = '3XMM_DR7cat_slim_v1.0.fits'
    
    ### Define and create (if needed) data folders
    root_folder = os.path.realpath(__file__)
    root_folder = os.path.dirname(root_folder)

    data_folder = os.path.join(root_folder, '../data')
    data_folder = os.path.normpath(data_folder)
    if not os.path.exists(data_folder) :
        os.makedirs(data_folder)

    xmm_folder  = os.path.join(data_folder, '3xmmdr7')
    if not os.path.exists(xmm_folder) :
        os.makedirs(xmm_folder)
    
    pstarrs_folder = os.path.join(data_folder, 'pstarrs1')
    if not os.path.exists(pstarrs_folder) :
        os.makedirs(pstarrs_folder)

    wise_folder = os.path.join(data_folder, 'allwise')
    if not os.path.exists(wise_folder) :
        os.makedirs(wise_folder)

    nir_folder = os.path.join(data_folder, nir_survey.lower())
    if not os.path.exists(nir_folder) :
        os.makedirs(nir_folder)

    if nir_survey is '2MASS' :
        nir_label = 'NTM'
        url_moc = None
        errtype_nir = 'ellipse'
        

    elif nir_survey is 'UKIDSS' :
        nir_label = 'NUK'
        url_moc = 'http://horus.roe.ac.uk/vsa/coverage-maps/UKIDSS/DR10/'
        errtype_nir = 'rcd_dec_ellipse'

    elif nir_survey is 'VISTA' :
        nir_label = 'NVT'
        url_moc = 'http://horus.roe.ac.uk/vsa/coverage-maps/VISTA/VHS/'
        errtype_nir = 'circle'
        
    else :
        raise ValueError('Invalid near-infrared survey!')
        
    nir_moc = utils.get_moc(url_moc, nir_survey, nir_folder)

    xmatch_folder = os.path.join(root_folder, '../xmatch')
    xmatch_folder = os.path.normpath(xmatch_folder)
    if not os.path.exists(xmatch_folder) :
        os.makedirs(xmatch_folder)

    folders_dict = {'xmatch': xmatch_folder, 'xmm': xmm_folder,
                    'pstarrs': pstarrs_folder, 'wise': wise_folder,
                    'nir': nir_folder}

    ### Get the list of XMM-Newton observations in the XMM catalogue
    xmmobsids_file_org = os.path.join(xmm_folder, obs_filename)

    if not os.path.isfile(xmmobsids_file_org) :
        utils.downloadFile(os.path.join(cat_url, obs_filename), 
                           folders_dict['xmm'])
        
    ### Select valid obsids (clean observations and in Pan-STARRS footprint)
    # Open table
    try :
        xmmobs = Table.read(xmmobsids_file_org)
        
    except :
        print("Unable to open XMM OBSIDs table!!!")
        print(xmmobsids_file_org)
        return


    # Select clean observations
    msk_obsclass = xmmobs.columns['OBS_CLASS'] < 4
    xmmobs_clean = xmmobs[msk_obsclass]
    
    # Select observations in Pan-STARRS footprint (dec>-30. deg)   
    msk_pstarrs = xmmobs_clean.columns['DEC'] > -30 + radius.to(u.deg).value    
    xmmobs_clean_pstarrs = xmmobs_clean[msk_pstarrs][:300]
   
    
    ### Define non-overlapping mocs for the obsids    
    if make_mocs :    
        xmmmocs.make_mocs(xmmobs_clean_pstarrs, folders_dict['xmm'], 
                          moc_order=16, radius=radius, remove_stars=True)


    ### Get data for the cross-match
    ## X-rays
    # Check if the XMM sources catalogue exists and download otherwise
    xmmcat_file = os.path.join(folders_dict['xmm'], src_filename)

    if not os.path.isfile(xmmcat_file) :
        src_filename_gz = '{}.gz'.format(src_filename)
        utils.downloadFile(os.path.join(cat_url, src_filename_gz), 
                           folders_dict['xmm'])
        utils.gunzipFile(os.path.join(folders_dict['xmm'], src_filename_gz), 
                         xmmcat_file)

    # Correct astrometry of XMM sources
    if poscorr :
        # Check if the detections catalogue exists and download otherwise
        xmmdet_file = os.path.join(folders_dict['xmm'], det_filename)
        
        if not os.path.isfile(xmmdet_file) :
            det_filename_gz = '{}.gz'.format(det_filename)            
            utils.downloadFile(os.path.join(cat_url, det_filename_gz), 
                               folders_dict['xmm'])
            utils.gunzipFile(os.path.join(folders_dict['xmm'], det_filename_gz), 
                             xmmdet_file)

        xposcorr.run(xmmdet_file, xmmcat_file, xmmobs_clean_pstarrs, 
                     folders_dict['xmm'])
        
        return
    
    # Make files with X-ray sources per non-overlaping field
    file_name, file_ext = os.path.splitext(xmmobsids_file_org)
    xmmobsids_file = '{}_{}_clean_pstarrs{}'.format(file_name, nir_survey.lower(),
                                                    file_ext)

    if getXdata :
        xmmobs_xdata = getdata.xmm(xmmobs_clean_pstarrs, folders_dict['xmm'], 
                                   xmmcat_file, nir_moc=nir_moc, moc_order=16, 
                                   use_poscorr=False)

        # Save selected obsids 
        # (with Texp and sky area, remove fields with no sources)
        xmmobs_xdata.write(xmmobsids_file, overwrite=True)

    else :
        xmmobs_xdata = Table.read(xmmobsids_file)

    ## Pan-STARRS
    file_name, file_ext = os.path.splitext(xmmobsids_file)
    xmmobsids_file = '{}_nPSsrc{}'.format(file_name, file_ext)
    
    if getPSdata :
        xmmobs_psdata = getdata.pstarrs(xmmobs_xdata, folders_dict['pstarrs'], 
                                        xmm_folder, nir_moc=nir_moc, moc_order=16, 
                                        radius=radius, overwrite=False)
                                        
        xmmobs_psdata.write(xmmobsids_file, overwrite=True)

    else :
        xmmobs_psdata = Table.read(xmmobsids_file)
    
    ## All-WISE
    file_name, file_ext = os.path.splitext(xmmobsids_file)
    xmmobsids_file = '{}_nWSsrc{}'.format(file_name, file_ext)
    
    if getWSdata :
        xmmobs_wsdata = getdata.wise(xmmobs_psdata, folders_dict['wise'], 
                                     xmm_folder, nir_moc=nir_moc, moc_order=16, 
                                     radius=radius, overwrite=False)
                                        
        xmmobs_wsdata.write(xmmobsids_file, overwrite=True)

    else :
        xmmobs_wsdata = Table.read(xmmobsids_file)
       
    ## NIR data
    file_name, file_ext = os.path.splitext(xmmobsids_file)
    xmmobsids_file = '{}_n{}src{}'.format(file_name, nir_label, file_ext)
       
    if getNIRdata :
        if nir_survey is '2MASS' :
            xmmobs_nirdata = getdata.tmass(xmmobs_wsdata, folders_dict['nir'], 
                                           xmm_folder, moc_order=16, 
                                           radius=radius, overwrite=False)
        elif nir_survey is 'UKIDSS' :
            xmmobs_nirdata = getdata.ukidss(xmmobs_wsdata, folders_dict['nir'], 
                                            xmm_folder, moc_order=16, 
                                            radius=radius, overwrite=True)
        elif nir_survey is 'VISTA' :
            xmmobs_nirdata = getdata.vista(xmmobs_wsdata, folders_dict['nir'], 
                                           xmm_folder, moc_order=16, 
                                           radius=radius, overwrite=True)

        xmmobs_nirdata.write(xmmobsids_file, overwrite=True)

    else :
        xmmobs_nirdata = Table.read(xmmobsids_file)
    

    ### Calculate bins according to XMM exposure time and galactic latitude
    file_name, file_ext = os.path.splitext(xmmobsids_file_org)
    xmmobsids_file = '{}_{}_bins{}'.format(file_name, nir_survey.lower(), 
                                           file_ext)

    if define_bins :
        ## Galactic latitude binning
        xmmobs_optbin = binning.optical(xmmobs_nirdata, data_folder, nir_survey)
        
        ## XMM exposure binning
        xmmobs_bins = binning.final(xmmobs_optbin, data_folder, nir_survey)
        xmmobs_bins.write(xmmobsids_file, overwrite=True)
    
    else :
        xmmobs_bins = Table.read(xmmobsids_file)


    ### Make bins
    if make_bins :
        binning.makebins(xmmobs_bins, folders_dict['xmm'], 'XMM', 
                         nir_survey, errtype='circle')
        binning.makebins(xmmobs_bins, folders_dict['pstarrs'], 'Pan-STARRS', 
                         nir_survey, errtype='rcd_dec_ellipse')
        binning.makebins(xmmobs_bins, folders_dict['wise'], 'WISE', 
                         nir_survey, errtype='ellipse')
        binning.makebins(xmmobs_bins, folders_dict['nir'], nir_survey, 
                         errtype=errtype_nir)
    
    ### Crossmatching of catalogues
    xmatchcat_filename = '{}_xmatchcat.fits'.format(nir_survey.lower())
    xmatchcat_filename = os.path.join(folders_dict['xmatch'], xmatchcat_filename)
    
    if make_xmatch :
        stats_filename = '{}_bins.fits'.format(nir_survey.lower())
        bin_stats = Table.read(os.path.join(data_folder, stats_filename))
        
        crossmatching.run(bin_stats, folders_dict, nir_survey, nir_label)
        crossmatching.merge_bins(bin_stats, folders_dict, nir_survey)
        xmatch_cat = crossmatching.merge_cats(folders_dict, nir_survey, nir_label)
        xmatch_cat.write(xmatchcat_filename, overwrite=True)

    else :
        xmatch_cat = Table.read(xmatchcat_filename, memmap=True)

    return xmatch_cat