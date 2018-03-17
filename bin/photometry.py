# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:45:39 2018

@author: alnoah
"""

import os
import requests
import numpy as np
import time

from bs4 import BeautifulSoup
from io import BytesIO
from tqdm import trange

from astropy.table import Table, join, unique, vstack, hstack
from astropy.coordinates import SkyCoord
from astroquery.irsa_dust import IrsaDust

import casjobs


def pstarrs(srcids):

    casjobs.add_table(Table([srcids]), 'xmatch_objids')

    filters = ['g', 'r', 'i', 'z', 'y']
    cols_mflags = ''.join(['m.{}Flags, '.format(f) for f in filters])
    cols_mPSFMag = ''.join(['m.{0}MeanPSFMag, m.{0}MeanPSFMagErr, '.format(f) 
                            for f in filters])
    cols_mKronMag = ''.join(['m.{0}MeanKronMag, m.{0}MeanKronMagErr, '.format(f) 
                            for f in filters])
    cols_mApMag = 'm.rMeanApMag, m.rMeanApMagErr, m.iMeanApMag, m.iMeanApMagErr'

    cols_infoflags = ''.join(['t.{}infoFlag, '.format(f) for f in filters])
    cols_PSFMag = ''.join(['t.{0}PSFMag, t.{0}PSFMagErr, '.format(f) 
                            for f in filters])
    cols_KronMag = ''.join(['t.{0}KronMag, t.{0}KronMagErr, '.format(f) 
                            for f in filters])
    cols_ApMag = 't.rApMag, t.rApMagErr, t.iApMag, t.iApMagErr'    
    
    # Get stack photometry
    cols = 't.objid, t.bestDetection, {}{}{}{}'
    cols = cols.format(cols_infoflags, cols_PSFMag, cols_KronMag, cols_ApMag)

    qry = 'select {} from MyDB.xmatch_objids s '.format(cols)
    qry += 'inner join StackObjectThin t on s.PSobjID=t.objid '
    qry += 'where t.primaryDetection=1 '
    qry += 'into xmatch_objids_stackPhoto'
    casjobs.run_qry(qry, 'stackPhoto')

    casjobs.get_table('xmatch_objids_stackPhoto')
    
    photo_stack = Table.read('xmatch_objids_stackPhoto.fits', memmap=True)

    # remove duplicate sources
    photo_stack.add_column(-photo_stack['rPSFMag'], name='mr')
    photo_stack.sort('mr')
    photo_stack.remove_column('mr')    
    photo_stack = unique(photo_stack, keys='objid')

    # Get mean photometry
    cols = 'o.objid, o.raMean, o.decMean, o.surveyID, '
    cols += 'o.objInfoFlag, o.qualityFlag, {}{}{}{}'
    cols = cols.format(cols_mflags, cols_mPSFMag, cols_mKronMag, cols_mApMag)
    
    qry = 'select {} from MyDB.xmatch_objids s '.format(cols)
    qry += 'inner join ObjectThin o on o.objid=s.PSobjID '
    qry += 'inner join MeanObject m on o.objid=m.objid and o.uniquePspsOBid=m.uniquePspsOBid '
    qry += 'into xmatch_objids_meanPhoto'
    casjobs.run_qry(qry, 'meanPhoto')

    casjobs.get_table('xmatch_objids_meanPhoto')
    
    photo_mean = Table.read('xmatch_objids_meanPhoto.fits', memmap=True)

    # Clean temp files
    casjobs.drop_table('xmatch_objids_stackPhoto')
    casjobs.drop_table('xmatch_objids_meanPhoto')    
    casjobs.drop_table('xmatch_objids')
    os.remove('xmatch_objids_meanPhoto.fits')
    os.remove('xmatch_objids_stackPhoto.fits')
    
    photo = join(photo_mean, photo_stack, join_type='left', keys='objid')
    
    return photo


def pstarrs_clean(photo):
    
    filters = ['g', 'r', 'i', 'z', 'y']
    msk_ptl = (photo['objInfoFlag'] & 0x00800000) == 0
    
    good_stack = np.zeros((len(photo), len(filters)))
    good_mean = np.zeros((len(photo), len(filters)))
    
    for i, f in enumerate(filters) :
        stack_colname = 'stack_{}Mag'.format(f)
        mean_colname = 'mean_{}Mag'.format(f)
        
        mag = photo[f + 'KronMag']
        mag[msk_ptl] = photo[msk_ptl][f + 'PSFMag']
        magErr = photo[f + 'KronMagErr']
        magErr[msk_ptl] = photo[msk_ptl][f + 'PSFMagErr']
                
        photo.add_column(Table.Column(mag), name=stack_colname)
        photo.add_column(Table.Column(magErr), name=stack_colname + 'Err')

        mag = photo[f + 'MeanKronMag']
        mag[msk_ptl] = photo[msk_ptl][f + 'MeanPSFMag']
        magErr = photo[f + 'MeanKronMagErr']
        magErr[msk_ptl] = photo[msk_ptl][f + 'MeanPSFMagErr']
                
        photo.add_column(Table.Column(mag), name=mean_colname)
        photo.add_column(Table.Column(magErr), name=mean_colname + 'Err')
        
        msk_good = np.logical_and(photo[stack_colname] > 0,
                                  photo[stack_colname + 'Err'] > 0)        
        good_stack[msk_good,i] = [1]*len(np.where(msk_good))

        msk_good = np.logical_and(photo[mean_colname] > 0,
                                  photo[mean_colname + 'Err'] > 0)        
        good_mean[msk_good,i] = [1]*len(np.where(msk_good))
    
    n_stack = np.sum(good_stack, axis=1)
    n_mean = np.sum(good_mean, axis=1)
    msk_stack = np.logical_and(~photo['bestDetection'].mask,
                               n_stack >= n_mean)
    photo.add_column(Table.Column(msk_stack), name='stack')
    
    for f in filters :
        colname = '{}Mag'.format(f)
        
        mag = photo['mean_{}Mag'.format(f)]
        mag[msk_stack] = photo[msk_stack]['stack_{}Mag'.format(f)]
        magErr = photo['mean_{}MagErr'.format(f)]
        magErr[msk_stack] = photo[msk_stack]['stack_{}MagErr'.format(f)]

        photo.add_column(Table.Column(mag), name=colname)
        photo.add_column(Table.Column(magErr), name=colname + 'Err')

    photo.keep_columns(['objid', 'objInfoFlag', 'qualityFlag', 'stack',
                        'gMag', 'rMag', 'iMag', 'zMag', 'yMag', 
                        'gMagErr', 'rMagErr', 'iMagErr', 'zMagErr', 'yMagErr'])
    
    return photo

def query_wsa(srcids, columns='ra,dec', constraints='sourceID=u.NIRobjID', 
              table='lasSource', database='UKIDSSDR10PLUS',
              url='http://wsa.roe.ac.uk:8080/wsa/WSASQL'):

    Table([srcids]).write('temp.fits', format='fits', overwrite=True)

    qry = 'SELECT {} FROM {}, #userTable as u WHERE {}'
    qry = qry.format(columns, table, constraints)
    payload = {'database': database, 
               'formaction': 'freeform',
               'uploadSQLFile': '',
               'sqlstmt': qry,
               'iFmt': 'FITS',
               'emailAddress': '', 
               'format': 'FITS',
               'compress': 'NONE',
               'rows': 2,
               'timeout': 10800}
    file = {'uploadFileToTable': open('temp.fits','rb')}
    r = requests.post(url, files=file, data=payload)
    os.remove('temp.fits')    
    
    soup = BeautifulSoup(r.text, "html5lib")
    dl = soup.find('a', id='dl_id')
    r = requests.get(dl['href'])
    
    return Table.read(BytesIO(r.content), format='fits')
    
def wise(srcids, columns='cntr,w1mpro,w2mpro'):

    columns = ''.join(['w.{}, '.format(s.replace(' ', '')) 
                       for s in columns.split(',')]).rstrip(', ')

    photo = query_wsa(srcids, columns=columns, constraints='w.cntr=u.WSID',
                      table='WISE..allwise_sc as w')
    return photo

def tmass(srcids, columns='cntr,w1mpro,w2mpro'):

    columns = ''.join(['t.{}, '.format(s.replace(' ', '')) 
                       for s in columns.split(',')]).rstrip(', ')

    photo = query_wsa(srcids, columns=columns, constraints='t.pts_key=u.NIRobjID',
                      table='TWOMASS..twomass_psc as t')    
    return photo

def ukidss(srcids, columns='ra,dec', table='lasSource'):

    photo = query_wsa(srcids, columns=columns, 
                      constraints='sourceID=u.NIRobjID', 
                      table=table)
    return photo

def vista(srcids, columns='ra,dec', table='vhsSource'):

    photo = query_wsa(srcids, columns=columns, 
                      constraints='sourceID=u.NIRobjID',
                      table=table, database='VHSDR4', 
                      url='http://horus.roe.ac.uk:8080/vdfs/WSASQL')
    return photo

def merge(catalogue, photo_ps, photo_ws, photo_tm, photo_uk, photo_vt):
    
    # Pan-STARRS photometry
    photo_ps.rename_column('objid', 'PSobjID')
    photo = join(catalogue, photo_ps, join_type='left', keys='PSobjID')

    # NIR photometry
    photo_nir_tm = Table([photo_tm['PTS_KEY'], 
                          photo_tm['J_M'], photo_tm['J_MSIGCOM'], 
                          photo_tm['H_M'], photo_tm['H_MSIGCOM'],
                          photo_tm['K_M'], photo_tm['K_MSIGCOM'],
                          photo_tm['CC_FLG'], photo_tm['PH_QUAL']],
                         names=['NIRobjID',
                               'JMag', 'JMagErr', 'HMag', 'HMagErr', 
                               'KMag', 'KMagErr', 'nir_cc_flags', 'nir_ph_qual'])

    photo_nir_uk = Table([photo_uk['SOURCEID'], photo_uk['J_1PPERRBITS'], 
                          photo_uk['HPPERRBITS'], photo_uk['KPPERRBITS'],
                          photo_uk['J_1APERMAG3'], photo_uk['J_1APERMAG3ERR'], 
                          photo_uk['HAPERMAG3'], photo_uk['HAPERMAG3ERR'],
                          photo_uk['KAPERMAG3'], photo_uk['KAPERMAG3ERR']],
                         names=['NIRobjID',
                                'Jpperrbit', 'Hpperrbit', 'Kpperrbit',
                                'JMag', 'JMagErr', 'HMag', 'HMagErr', 
                                'KMag', 'KMagErr'])

    photo_nir_vt = Table([photo_vt['SOURCEID'], photo_vt['JPPERRBITS'], 
                          photo_vt['HPPERRBITS'], photo_vt['KSPPERRBITS'],
                          photo_vt['JAPERMAG3'], photo_vt['JAPERMAG3ERR'], 
                          photo_vt['HAPERMAG3'], photo_vt['HAPERMAG3ERR'],
                          photo_vt['KSAPERMAG3'], photo_vt['KSAPERMAG3ERR']],
                         names=['NIRobjID',
                                'Jpperrbit', 'Hpperrbit', 'Kpperrbit',
                                'JMag', 'JMagErr', 'HMag', 'HMagErr', 
                                'KMag', 'KMagErr'])

    photo_nir = vstack([photo_nir_tm, photo_nir_uk, photo_nir_vt])
    
    photo_goodnir = photo[~photo['NIRobjID'].mask]
    photo_badnir = photo[photo['NIRobjID'].mask]
    photo = join(photo_goodnir, photo_nir, join_type='left', keys='NIRobjID')
    photo = vstack([photo, photo_badnir])

    # WISE photometry
    photo_ws.rename_column('CNTR', 'WSID')
    photo_ws.rename_column('W1MPRO', 'W1Mag')
    photo_ws.rename_column('W1SIGMPRO', 'W1MagErr')
    photo_ws.rename_column('W2MPRO', 'W2Mag')
    photo_ws.rename_column('W2SIGMPRO', 'W2MagErr')
    photo_ws.rename_column('CC_FLAGS', 'ws_cc_flags')
    photo_ws.rename_column('PH_QUAL', 'ws_ph_qual')
    photo_ws.remove_columns(['W1SNR', 'W2SNR'])
    
    photo_goodws = photo[~photo['WSID'].mask]
    photo_badws = photo[photo['WSID'].mask]
    photo = join(photo_goodws, photo_ws, join_type='left', keys='WSID')
    photo = vstack([photo, photo_badws])
    
    return photo

def add_extinction(catalogue, getOptical=True, getIR=True, 
                   url_ned='http://ned.ipac.caltech.edu/cgi-bin/calc'):

    nsrcs = len(catalogue)
    A = np.full((nsrcs,10), np.nan)
    coords = SkyCoord(ra=catalogue['posRA'], dec=catalogue['posDec'])
    
    start=0
    for i in trange(nsrcs) :
        # Get optical extinction from NED
        if getOptical :
            payload = {'in_csys': 'Equatorial', 
                       'in_equinox': 'J2000.0', 
                       'obs_epoch': '2000.0', 
                       'out_csys': 'Equatorial', 
                       'out_equinox': 'J2000.0', 
                       'lon': '{:.6f}d'.format(coords[i].ra.value), 
                       'lat': '{:.6f}d'.format(coords[i].dec.value)}

            end = time.time()
            if end-start < 1.0 :
                time.sleep(1.0)
            else :
                r = requests.get(url_ned, params=payload)
            start = time.time()

            soup = BeautifulSoup(r.text, 'html5lib')
            ned_table = soup.find('div', id='moreBANDS')
            ned_table = ned_table.find('table')

            j = 0
            for row in ned_table.findAll("tr"):
                cells = row.findAll("td")
                if len(cells) :
                    if cells[0].contents[0] == 'PS1' and j < 5:
                        A[i,j] = cells[3].contents[0]
                        j += 1

        if getIR :
            t = IrsaDust.get_extinction_table(coords[i])
    
            # NIR extinction
            if catalogue['NIRobjID'][i] > 0:
                if catalogue['NIR_SURVEY'][i] == '2MASS':
                    A[i,5:8] = t['A_SandF'][16:19]
                
                elif (catalogue['NIR_SURVEY'][i] == 'UKIDSS' or 
                      catalogue['NIR_SURVEY'][i]=='VISTA'):
                    A[i,5:8] = t['A_SandF'][13:16]
            
            # WISE extinction
            if catalogue['WSID'][i] > 0:
                A[i,8:] = t['A_SandF'][23:25]

    A = Table(A, names=['Ag', 'Ar', 'Ai', 'Az', 'Ay', 
                        'AJ', 'AH', 'AK', 'AW1', 'AW2'])
        
    catalogue = hstack([catalogue, A], join_type='exact') 

    return catalogue

def define_samples(cat):

    filledcat = cat.filled(-999.)
    # good_opt
    msk_g = np.logical_and(filledcat['gMag'] > 0, filledcat['gMagErr'] > 0)
    msk_r = np.logical_and(filledcat['rMag'] > 0, filledcat['rMagErr'] > 0)
    msk_i = np.logical_and(filledcat['iMag'] > 0, filledcat['iMagErr'] > 0)
    msk_z = np.logical_and(filledcat['zMag'] > 0, filledcat['zMagErr'] > 0)
    msk_y = np.logical_and(filledcat['yMag'] > 0, filledcat['yMagErr'] > 0)    
    msk_opt = np.logical_and(np.logical_and(msk_g, msk_r), 
                             np.logical_and(msk_i, msk_z))
    msk_opt = np.logical_and(msk_opt, msk_y)
    
    # good nir
    msk_j = np.logical_and(filledcat['JMag'] > 0, filledcat['JMagErr'] > 0)
    msk_h = np.logical_and(filledcat['HMag'] > 0, filledcat['HMagErr'] > 0)
    msk_k = np.logical_and(filledcat['KMag'] > 0, filledcat['KMagErr'] > 0)
    msk_nir = np.logical_and(msk_j, msk_h)
    
    # good_mir
    msk_w1 = np.logical_and(filledcat['W1Mag'] > 0, filledcat['W1MagErr'] > 0)
    msk_w2 = np.logical_and(filledcat['W2Mag'] > 0, filledcat['W2MagErr'] > 0)
    msk_mir = np.logical_and(msk_w1, msk_w2)

    sample_10filters = np.logical_and(np.logical_and(msk_opt, msk_mir), 
                                      np.logical_and(msk_nir, msk_k))
    sample_10filters = np.logical_and(sample_10filters, cat['sampleXPWN'])
    
    sample_7filters = np.logical_and(np.logical_or(cat['sampleXPWN'], 
                                                   cat['sampleXPW']), 
                                     ~sample_10filters)
    sample_7filters = np.logical_and(np.logical_and(msk_opt, msk_mir), 
                                     sample_7filters)

    sample_8filters = np.logical_and(np.logical_or(cat['sampleXPWN'], 
                                                   cat['sampleXPN']), 
                                     np.logical_and(~sample_10filters, 
                                                   ~sample_7filters))
    sample_8filters = np.logical_and(np.logical_and(msk_opt, msk_nir), 
                                     sample_8filters)

    sample_5filters = np.logical_or(np.logical_or(sample_7filters, 
                                                  sample_8filters), 
                                    sample_10filters)
    sample_5filters = np.logical_and(msk_opt, ~sample_5filters)
    
    samples = Table([msk_g, msk_r, msk_i, msk_z, msk_y, 
                     msk_j, msk_h, msk_k, msk_w1, msk_w2,
                     msk_opt, msk_nir, msk_mir, 
                     sample_5filters, sample_7filters, 
                     sample_8filters, sample_10filters],
                    names=['good_g', 'good_r', 'good_i', 'good_z', 'good_y', 
                           'good_J', 'good_H', 'good_K', 'good_W1', 'good_W2', 
                           'good_opt', 'good_nir', 'good_mir', 
                           'sample_5mags', 'sample_7mags', 
                           'sample_8mags', 'sample_10mags'])

    cat = hstack([cat, samples], join_type='exact') 

    return cat
    
    
def add_data(catalogue, getPSphoto=False, getWSphoto=False, getUKphoto=False,
             getVTphoto=False, getTMphoto=False, extinction=False):

    data_folder = '../data/'    
    pstarrs_folder = os.path.join(data_folder, 'pstarrs1')
    wise_folder = os.path.join(data_folder, 'allwise')
    tmass_folder = os.path.join(data_folder, '2mass')
    ukidss_folder = os.path.join(data_folder, 'ukidss')
    vista_folder = os.path.join(data_folder, 'vista')
    
    # Select only sources in the photoz sample
    catalogue = catalogue[catalogue['sample_photoz']]
    
    photo_ps_file = os.path.join(pstarrs_folder, 'photometry.fits')
    if getPSphoto :
        photo_ps = pstarrs(catalogue['PSobjID'])
        
        root, ext = os.path.splitext(photo_ps_file)
        photo_ps.write('{}_all{}'.format(root,ext), overwrite=True)
        
        photo_ps = pstarrs_clean(photo_ps)
        photo_ps.write(photo_ps_file, overwrite=True)

    else :
        photo_ps = Table.read(photo_ps_file, memmap=True)
    
    photo_ws_file = os.path.join(wise_folder, 'photometry.fits')
    if getWSphoto :
        cols = 'cntr, cc_flags, ph_qual, '
        cols += 'w1mpro, w1sigmpro, w1snr, '
        cols += 'w2mpro, w2sigmpro, w2snr'

        msk = ~catalogue['WSID'].mask                 
        photo_ws = wise(catalogue[msk]['WSID'], columns=cols)

        photo_ws.write(photo_ws_file, overwrite=True)

    else :
        photo_ws = Table.read(photo_ws_file, memmap=True)
    
    photo_uk_file = os.path.join(ukidss_folder, 'photometry.fits')
    if getUKphoto :
        cols = 'sourceID, mergedClass, '
        cols += 'j_1AperMag3, j_1AperMag3Err, j_1ppErrBits, '        
        cols += 'hAperMag3, hAperMag3Err, hppErrBits, '
        cols += 'kAperMag3, kAperMag3Err, kppErrBits'

        msk = np.logical_and(~catalogue['NIRobjID'].mask, 
                             catalogue['NIR_SURVEY'] == 'UKIDSS')
        photo_uk = ukidss(catalogue[msk]['NIRobjID'], columns=cols)

        photo_uk.write(photo_uk_file, overwrite=True)

    else :
        photo_uk = Table.read(photo_uk_file, memmap=True)
    
    photo_vt_file = os.path.join(vista_folder, 'photometry.fits')
    if getVTphoto :
        cols = 'sourceID, mergedClass, '
        cols += 'jAperMag3, jAperMag3Err, jppErrBits, '        
        cols += 'hAperMag3, hAperMag3Err, hppErrBits, '
        cols += 'ksAperMag3, ksAperMag3Err, ksppErrBits'

        msk = np.logical_and(~catalogue['NIRobjID'].mask, 
                             catalogue['NIR_SURVEY'] == 'VISTA')
        photo_vt = vista(catalogue[msk]['NIRobjID'], columns=cols)

        photo_vt.write(photo_vt_file, overwrite=True)

    else :
        photo_vt = Table.read(photo_vt_file, memmap=True)
    
    photo_tm_file = os.path.join(tmass_folder, 'photometry.fits')
    if getTMphoto :
        cols = 'pts_key, cc_flg, ph_qual, '
        cols += 'j_m, j_msigcom, j_snr, '
        cols += 'h_m, h_msigcom, h_snr, '
        cols += 'k_m, k_msigcom, k_snr'
        
        msk = np.logical_and(~catalogue['NIRobjID'].mask, 
                             catalogue['NIR_SURVEY'] == '2MASS')
        photo_tm = tmass(catalogue[msk]['NIRobjID'], columns=cols)

        photo_tm.write(photo_tm_file, overwrite=True)

    else :
        photo_tm = Table.read(photo_tm_file, memmap=True)

    catalogue_photo = merge(catalogue, photo_ps, photo_ws, photo_tm, photo_uk, photo_vt)
    if extinction :
        catalogue_photo = add_extinction(catalogue_photo)
    catalogue_photo = define_samples(catalogue_photo)    
    
    return catalogue_photo