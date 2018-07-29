# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:49:20 2018

@author: alnoah
"""

import os
import logging

import numpy as np
from astropy.table import Table, vstack, unique, join
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


def dir_structure(opt_survey, nir_survey):
    """
    Define and create (if needed) data folders
    """
    root_folder = os.path.realpath(__file__)
    root_folder = os.path.dirname(root_folder)

    data_folder = os.path.join(root_folder, '../{}_data'.format(opt_survey))
    data_folder = os.path.normpath(data_folder)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    xmm_folder  = os.path.join(data_folder, '3xmmdr8')
    if not os.path.exists(xmm_folder):
        os.makedirs(xmm_folder)
    
    opt_folder = os.path.join(data_folder, opt_survey)
    if not os.path.exists(opt_folder):
        os.makedirs(opt_folder)

    wise_folder = os.path.join(data_folder, 'allwise')
    if not os.path.exists(wise_folder):
        os.makedirs(wise_folder)

    nir_folder = os.path.join(data_folder, nir_survey.lower())
    if not os.path.exists(nir_folder):
        os.makedirs(nir_folder)

    xmatch_folder = os.path.join(root_folder, '../{}_xmatch'.format(opt_survey))
    xmatch_folder = os.path.normpath(xmatch_folder)
    if not os.path.exists(xmatch_folder):
        os.makedirs(xmatch_folder)

    folders_dict = {'xmatch': xmatch_folder, 'data': data_folder,
                    'xmm': xmm_folder, 'opt': opt_folder,
                    'wise': wise_folder, 'nir': nir_folder}

    return folders_dict


def clean_obsids(obsids, radius, opt_survey='pstarrs', moc=None):
    # Select clean observations
    xmmobs_clean = obsids[obsids.columns['OBS_CLASS'] < 4]
    
    if opt_survey == 'pstarrs':
        # Select observations in Pan-STARRS footprint (dec>-30. deg)   
        msk_pstarrs = xmmobs_clean.columns['DEC'] > -30 + radius.to(u.deg).value    
        xmmobs_clean_opt = xmmobs_clean[msk_pstarrs]

    elif opt_survey == 'sdss':
        # Select observations in SDSS footprint
        sdss_moc = MOC()
        read_moc_fits(sdss_moc, moc)
        moc_order = sdss_moc.order

        hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())
        xmmobs_clean_opt = utils.sources_inmoc(xmmobs_clean, hp, sdss_moc, 
                                               moc_order=moc_order, 
                                               ra='RA', dec='DEC', units=u.deg)        
    else:
        raise ValueError('Unknown optical survey: {}'.format(opt_survey))

    return xmmobs_clean_opt


def final_cat(probacat, photozcat):
    photozcat.remove_columns(['ztrue','zmean1', 'zConf1', 'err1'])
    photozcat.rename_column('id', 'PSobjID')
    photozcat.rename_column('zmode0', 'PHOT_Z')
    photozcat.rename_column('zConf0', 'PHOT_ZCONF')
    photozcat.rename_column('err0', 'PHOT_ZERR')
    
    finalcat = join(probacat, photozcat, join_type='right', keys=['PSobjID'])

    return finalcat

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
    msk = np.logical_and(xmatchcat_final['NIRobjID'].mask, msk)

    return xmatchcat_final


def make_cat(opt_survey, nir_survey, radius=15*u.arcmin, poscorr=False, 
             make_mocs=False, getXdata=False, getOPTdata=False, 
             getWSdata=False, getNIRdata=False, define_bins=False, 
             make_bins=False, make_xmatch=False):
        
    cat_url = 'http://xmmssc.irap.omp.eu/Catalogue/3XMM-DR8'
    obs_filename = '3xmmdr8_obslist.fits'
    det_filename = '3XMM_DR8cat_v1.0.fits'
    src_filename = '3XMM_DR8cat_slim_v1.0.fits'

    # Increase in radius to avoid border effects in the crossmatch:
    delta_radius = 0.3*u.arcmin # (18 arcsec, like ARCHES)

    # Define structure of data directories
    dirs = dir_structure(opt_survey, nir_survey)
 
    # Get moc for the optical survey footprint
    if opt_survey == 'pstarrs':
        opt_label = 'PS'
        opt_moc = None

    elif opt_survey == 'sdss':
        opt_label = 'SDSS'
        opt_moc = utils.get_moc(('http://alasky.unistra.fr/footprints/tables/'
                                 'vizier/V_139_sdss9/MOC?nside=2048'), 
                                opt_survey, dirs['opt'])
    else:
        raise ValueError('Unknown optical survey!')

    # Get moc for the nir survey footprint
    if nir_survey is '2MASS':
        nir_label = 'NTM'
        url_moc = None
        errtype_nir = 'ellipse'

    elif nir_survey is 'UKIDSS':
        nir_label = 'NUK'
        url_moc = 'http://horus.roe.ac.uk/vsa/coverage-maps/UKIDSS/DR10/'
        errtype_nir = 'rcd_dec_ellipse'

    elif nir_survey is 'VISTA':
        nir_label = 'NVT'
        url_moc = 'http://horus.roe.ac.uk/vsa/coverage-maps/VISTA/VHS/'
        errtype_nir = 'circle'
        
    else:
        raise ValueError('Unknown near-infrared survey!')
        
    nir_moc = utils.get_moc(url_moc, nir_survey, dirs['nir'])

    ### Get the list of XMM-Newton observations in the XMM catalogue
    xmmobsids_file_org = os.path.join(dirs['xmm'], obs_filename)

    if not os.path.isfile(xmmobsids_file_org):
        utils.downloadFile(os.path.join(cat_url, obs_filename), dirs['xmm'])
        
    ### Select valid obsids (clean observations and in the optical footprint)
    try:
        xmmobs = Table.read(xmmobsids_file_org)

    except:
        message = 'Unable to open XMM OBSIDs table!!!\nFile: {}'
        logging.error(message.format(xmmobsids_file_org))
        return
        
    xmmobs_clean_opt = clean_obsids(xmmobs, radius, opt_survey, opt_moc)

    ### Define non-overlapping mocs for the obsids    
    if make_mocs:
        xmmmocs.make_mocs(xmmobs_clean_opt, dirs['xmm'],
                          moc_order=15, radius=radius + delta_radius,
                          remove_stars=True, remove_large_galaxies=True)

    ### Get data for the cross-match
    ## X-rays
    # Check if the XMM sources catalogue exists, and download otherwise
    xmmcat_file = os.path.join(dirs['xmm'], src_filename)

    if not os.path.isfile(xmmcat_file):
        src_filename_gz = '{}.gz'.format(src_filename)
        utils.downloadFile(os.path.join(cat_url, src_filename_gz), dirs['xmm'])
        utils.gunzipFile(os.path.join(dirs['xmm'], src_filename_gz), 
                         xmmcat_file)

    # Correct astrometry of XMM sources
    if poscorr:
        # Check if the detections catalogue exists and download otherwise
        xmmdet_file = os.path.join(dirs['xmm'], det_filename)
        
        if not os.path.isfile(xmmdet_file):
            det_filename_gz = '{}.gz'.format(det_filename)            
            utils.downloadFile(os.path.join(cat_url, det_filename_gz), 
                               dirs['xmm'])
            utils.gunzipFile(os.path.join(dirs['xmm'], det_filename_gz), 
                             xmmdet_file)

        xposcorr.run(xmmdet_file, xmmcat_file, xmmobs_clean_opt, dirs['xmm'])
    
    # Make files with X-ray sources per non-overlaping field
    file_name, file_ext = os.path.splitext(xmmobsids_file_org)
    xmmobsids_file = '{}_{}_clean_{}{}'.format(file_name, nir_survey.lower(),
                                               opt_survey, file_ext)

    if getXdata:
        xmmobs_xdata = getdata.xmm(xmmobs_clean_opt, dirs['xmm'], xmmcat_file, 
                                   nir_moc=nir_moc, opt_moc=opt_moc,
                                   moc_order=15, radius=radius,
                                   use_poscorr=False)

        # Save selected obsids 
        # (with Texp and sky area, remove fields with no sources)
        xmmobs_xdata.write(xmmobsids_file, overwrite=True)

    else :
        xmmobs_xdata = Table.read(xmmobsids_file)

    ## Optical
    file_name, file_ext = os.path.splitext(xmmobsids_file)
    xmmobsids_file = '{}_n{}src{}'.format(file_name, opt_label, file_ext)
    
    if getOPTdata:
        if opt_survey == 'pstarrs':
            xmmobs_optdata = getdata.pstarrs(xmmobs_xdata, dirs['opt'], 
                                             dirs['xmm'], nir_moc=nir_moc, 
                                             radius=radius + delta_radius,
                                             moc_order=15, overwrite=False)
        elif opt_survey == 'sdss':
            xmmobs_optdata = getdata.sdss(xmmobs_xdata, dirs['opt'], 
                                          dirs['xmm'], nir_moc=nir_moc, 
                                          radius=radius + delta_radius,
                                          moc_order=15, overwrite=False)
        else:
            raise ValueError('Unknown optical survey!')

        xmmobs_optdata.write(xmmobsids_file, overwrite=True)

    else:
        xmmobs_optdata = Table.read(xmmobsids_file)
    
    ## All-WISE
    file_name, file_ext = os.path.splitext(xmmobsids_file)
    xmmobsids_file = '{}_nWSsrc{}'.format(file_name, file_ext)
    
    if getWSdata:
        xmmobs_wsdata = getdata.wise(xmmobs_optdata, dirs['wise'], dirs['xmm'], 
                                     nir_moc=nir_moc, opt_moc=opt_moc, 
                                     radius=radius + delta_radius,
                                     moc_order=15, overwrite=False)
                                        
        xmmobs_wsdata.write(xmmobsids_file, overwrite=True)

    else:
        xmmobs_wsdata = Table.read(xmmobsids_file)
       
    ## NIR data
    file_name, file_ext = os.path.splitext(xmmobsids_file)
    xmmobsids_file = '{}_n{}src{}'.format(file_name, nir_label, file_ext)
       
    if getNIRdata:
        if nir_survey is '2MASS':
            xmmobs_nirdata = getdata.tmass(xmmobs_wsdata, dirs['nir'], 
                                           dirs['xmm'], moc_order=15, 
                                           opt_moc=opt_moc,
                                           radius=radius + delta_radius, 
                                           overwrite=False)
        elif nir_survey is 'UKIDSS':
            xmmobs_nirdata = getdata.ukidss(xmmobs_wsdata, dirs['nir'], 
                                            dirs['xmm'], moc_order=15, 
                                            opt_moc=opt_moc,
                                            radius=radius + delta_radius, 
                                            overwrite=False)
        elif nir_survey is 'VISTA':
            xmmobs_nirdata = getdata.vista(xmmobs_wsdata, dirs['nir'], 
                                           dirs['xmm'], moc_order=15,
                                           opt_moc=opt_moc,
                                           radius=radius + delta_radius, 
                                           overwrite=False)

        xmmobs_nirdata.write(xmmobsids_file, overwrite=True)

    else:
        xmmobs_nirdata = Table.read(xmmobsids_file)
    

    ### Calculate bins according to XMM exposure time and galactic latitude
    file_name, file_ext = os.path.splitext(xmmobsids_file_org)
    xmmobsids_file = '{}_{}_bins{}'.format(file_name, nir_survey.lower(),
                                           file_ext)
    if define_bins:
        ## Galactic latitude binning
        xmmobs_optbin = binning.optical(xmmobs_nirdata, dirs['data'], 
                                        nir_survey, opt_survey)
        ## XMM exposure binning
        xmmobs_bins = binning.final(xmmobs_optbin, dirs['data'], nir_survey)
        xmmobs_bins.write(xmmobsids_file, overwrite=True)

    else:
        xmmobs_bins = Table.read(xmmobsids_file)


    ### Make bins
    if make_bins:
        binning.makebins(xmmobs_bins, dirs['xmm'], 'XMM', 
                         nir_survey, errtype='circle')
        binning.makebins(xmmobs_bins, dirs['opt'], opt_survey,
                         nir_survey, errtype='rcd_dec_ellipse')
        binning.makebins(xmmobs_bins, dirs['wise'], 'WISE',
                         nir_survey, errtype='ellipse')
        binning.makebins(xmmobs_bins, dirs['nir'], nir_survey,
                         errtype=errtype_nir)


    ### Crossmatching of catalogues
    xmatchcat_filename = '{}_xmatchcat.fits'.format(nir_survey.lower())
    xmatchcat_filename = os.path.join(dirs['xmatch'], xmatchcat_filename)

    if make_xmatch:
        stats_filename = '{}_bins.fits'.format(nir_survey.lower())
        bin_stats = Table.read(os.path.join(dirs['data'], stats_filename))
        
        crossmatching.run(bin_stats, dirs, opt_survey,
                          opt_label, nir_survey, nir_label)
        crossmatching.merge_bins(bin_stats, dirs, opt_survey, nir_survey)
        xmatch_cat = crossmatching.merge_cats(dirs, opt_survey, opt_label,
                                              nir_survey, nir_label)
        xmatch_cat.write(xmatchcat_filename, overwrite=True)

    else:
        xmatch_cat = Table.read(xmatchcat_filename, memmap=True)

    return xmatch_cat