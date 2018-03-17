# -*- coding: utf-8 -*-

# Correction of X-ray positions using Pan-STARRS sources
# and eposcorr SAS task


import os
import glob
import shutil
import subprocess as sp
import numpy as np

from tqdm import tqdm

from astropy.table import Table, join, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.vizier import Vizier

import utils


def eposcorr(xtable, otable, rotation='yes', 
             xra='RA', xdec='DEC', xradecerr='RADEC_ERR ', 
             ora='RAJ2000', odec='DEJ2000', oradecerr='POSERR') :
    """
    Parser for the SAS task eposcorr. It returns an astropy table with 
    the SRC_NUM of the X-ray sources and the corrected coordinates.
    """                 
    cmd = 'eposcorr xrayset={} opticalset={} '.format(xtable, otable)
    cmd += 'xrayra={} xraydec={} xrayradecerr={} '.format(xra, xdec, xradecerr)
    cmd += 'opticalra={} opticaldec={} opticalradecerr={} '.format(ora, odec, oradecerr)
    cmd += 'findrotation={} > /dev/null'.format(rotation)
    sp.call(cmd, shell=True)
    
    xtablecorr = Table.read(xtable)
    xtablecorr.keep_columns(['SRC_NUM', 'RA_CORR', 'DEC_CORR'])
    
    return xtablecorr


def getPPS(obsid) :
    """    
    Get PPS sources list for OBS_ID obsid.
    """
    url = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?'
    url += 'obsno={}&name=OBSMLI&level=PPS&extension=FTZ'.format(obsid)
    utils.downloadFile(url, '/tmp', 'temp.tar')
    
    try :
        utils.untarFile(os.path.join('/tmp','temp.tar'), '/tmp')
        xmm_srclist = glob.glob('/tmp/{0}/pps/P{0}EP*.FTZ'.format(obsid))

    except :
        # If untar fails, there is just one downloaded file 
        # and is probably the EPIC source list
        xmm_srclist = ['/tmp/temp.tar']
        
    return xmm_srclist[0]

    
def getPS(coords, v) :
    """
    Download Pan-STARRS sources in the field with coordinates coords, 
    an SkyCoord object, using Vizier object v.
    """
    vrsp = v.query_region(coords, radius=15*u.arcmin, catalog='II/349')
    
    pstarrs = vrsp[0]
    poserr = np.sqrt(pstarrs['e_RAJ2000']*pstarrs['e_DEJ2000'])
    pstarrs.add_column(poserr*u.arcsec, name='POSERR')        
    pstarrs.remove_columns(['e_RAJ2000', 'e_DEJ2000', 'Nd'])
    pstarrs.meta['description'] = ''
    pstarrs.meta['EXTNAME'] = 'RAWRES' # default extension name in eposcorr

    pstarrs_file = '/tmp/tmp.fits'        
    pstarrs.write(pstarrs_file, format='fits', overwrite=True)
    
    return pstarrs_file


def addCoordsCorr(detections, obsid, coordcorrs, filename) :
    """
    Add coordinate corrections to the list of detections in the field
    """
    detect_msk = detections['OBS_ID'] == obsid
    det_obsid = detections[detect_msk]
    det_obsid.remove_column('OBS_ID')

    det_obsid_corr = join(det_obsid, coordcorrs, keys='SRC_NUM')
    det_obsid_corr.remove_column('SRC_NUM')
    det_obsid_corr.meta = {}

    det_obsid_corr.write(filename, format='fits', overwrite=True)

    
def uniqueCoordsCorr(det_corrs_table) :
    """
    Calculates the unique source corrected coordinates SC_RA_CORR, SC_DEC_COOR
    in the astropy table det_corrs_table, using the mean RA_CORR and DEC_CORR of
    all detections, weighted by the positional errors POSERR.
    """
    srcids = np.unique(det_corrs_table['SRCID'])
    src_ra = np.full((len(srcids),), np.nan)
    src_dec = np.full((len(srcids),), np.nan)
    
    for i, srcid in enumerate(srcids) :
        msk = det_corrs_table['SRCID'] == srcid        
        det_srcid = det_corrs_table[msk]
        
        if len(det_srcid) > 1 :
            src_ra[i] = np.average(det_srcid['RA_CORR'], 
                                   weights=1/det_srcid['POSERR'])
            src_dec[i] = np.average(det_srcid['DEC_CORR'], 
                                    weights=1/det_srcid['POSERR'])
            
        else :
            src_ra[i] = det_srcid['RA_CORR']
            src_dec[i] = det_srcid['DEC_CORR']
            
    src_corrs = Table([srcids, src_ra*u.deg, src_dec*u.deg], 
                      names=['SRCID', 'SC_RA_CORR', 'SC_DEC_CORR'])
    
    return src_corrs


def run(detections, unqsrcs, obsids, data_folder) :
    """
    Position correction of XMM sources in the catalogue unqsrcs using the 
    SAS task eposcorr and Pan-STARRS optical sources. SAS environment must
    be initialized before running.
    """   
    poscorr_folder = os.path.join(data_folder, 'det_poscorr')
    if not os.path.exists(poscorr_folder) :
        os.makedirs(poscorr_folder)
    
    det_table = Table.read(detections, memmap=True)
    det_table.keep_columns(['SRCID', 'SRC_NUM', 'OBS_ID', 'RA', 'DEC', 'POSERR'])

    field_coords = SkyCoord(ra=obsids['RA']*u.deg, dec=obsids['DEC']*u.deg)
    obsids.keep_columns(['OBS_ID'])
    
    v = Vizier(columns=['RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000', 'Nd'], 
               column_filters={"Nd":">1"}, row_limit=np.inf, timeout=6000)
    
    tables_array = [None]*len(obsids)
    
    for i, obsid in enumerate(tqdm(obsids, desc='Correcting positions')) :
        #Get PPS sources list
        xmm_srclist = getPPS(obsid['OBS_ID'])
        
        # Get Pan-STARRS sources                       
        pstarrs_file = getPS(field_coords[i], v)
        
        # Correct positions using SAS
        xmm_srclist = eposcorr(xmm_srclist, pstarrs_file)        

        # Add corrected coords to the detections in the catalogue
        xmm_file = os.path.join(poscorr_folder, '{}.fits'.format(obsid['OBS_ID']))
        addCoordsCorr(det_table, obsid['OBS_ID'], xmm_srclist, xmm_file)
        
        tables_array[i] = Table.read(xmm_file, memmap=True)
        
        try :
            shutil.rmtree('/tmp/{}'.format(obsid['OBS_ID']))
        except :
            continue
        

    os.remove('/tmp/temp.tar')
    os.remove(pstarrs_file)
    del det_table
    
    # Coordinate corrections for unique sources
    det_corrs = vstack(tables_array)
    src_corrs = uniqueCoordsCorr(det_corrs)

    # Add corrections to the unique sources catalogue
    src_cat = Table.read(unqsrcs)
    src_cat_corrs = join(src_cat, src_corrs, keys='SRCID')
    
    root, ext = os.path.splitext(unqsrcs)
    filename = '{}_coordcorr{}'.format(root,ext)
    src_cat_corrs.write(filename, format='fits', overwrite=True)
    
    