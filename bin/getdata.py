# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm

from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS
from astropy import units as u
from astropy_healpix import HEALPix

from astroquery.vizier import Vizier
from astroquery.ukidss import Ukidss
from astroquery.vista import Vista
from astroquery.sdss import SDSS

from pymoc import MOC
from pymoc.io.fits import read_moc_fits

from utils import sources_inmoc


def xmm(obsids_table, data_folder, src_filename, nir_moc=None, opt_moc=None,
        moc_order=16, radius=15*u.arcmin, use_poscorr=False):
    """
    Get XMM sources
    For each observation in obsids_table, saves a fits file with
    name 'OBS_ID.fits' in 'data_folder/groups'.

    The function selects all sources in the catalogue 'src_filename' within
    'radius' arcmin of the RA, DEC of the observation, then it filters the
    result selecting the sources in the corresponding MOC stored in
    'moc_folder/mocs' (moc_order must be consistent with the order used to
    calculate the MOC).

    If use_poscorr is True, it uses the corrected coordinates of the catalogue
    (SAS task eposcorr and Pan-STARRS sources)

    The function returns obsids_table with additional columns 'TEXP_EP'
    (exposure time of the EPIC observation), 'SKY_AREA' (non-overlaping area
    of the observation) and 'NSRC_XMM' (number of X-ray sources in the field).
    """

    # Groups folder
    if nir_moc is None:
        groups_folder = os.path.join(data_folder, 'groups')
    else:
        root, _ = os.path.splitext(os.path.basename(nir_moc))
        survey = root.split('_')[-1]
        groups_folder = os.path.join(data_folder, 'groups_' + survey)

        moc_nirsurvey = MOC()
        read_moc_fits(moc_nirsurvey, nir_moc)

    if not os.path.exists(groups_folder):
        os.makedirs(groups_folder)

    if opt_moc is not None:
        moc_optsurvey = MOC()
        read_moc_fits(moc_optsurvey, opt_moc)

    src_table_all = Table.read(src_filename)
    msk_badsrc = src_table_all['SC_SUM_FLAG'] < 2

    if use_poscorr:
        rakey = 'SC_RA_CORR'
        deckey = 'SC_DEC_CORR'
    else:
        rakey = 'SC_RA'
        deckey = 'SC_DEC'

    src_table = src_table_all[msk_badsrc]
    src_catalog = SkyCoord(ra=src_table[rakey], dec=src_table[deckey])

    area_field = np.full((len(obsids_table),), np.nan)
    nsources_field = np.full((len(obsids_table),), np.nan)
    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())

    moc_folder = os.path.join(data_folder, 'mocs')

    for i, row in enumerate(tqdm(obsids_table, desc="Making XMM groups")):
        ## Load moc of the non-overlaping area
        moc_file = os.path.join(moc_folder, '{}.moc'.format(row['OBS_ID']))

        moc_field = MOC()
        read_moc_fits(moc_field, moc_file)

        if opt_moc is not None:
            moc_field = moc_optsurvey.intersection(moc_field)

        if nir_moc is not None:
            moc_field = moc_nirsurvey.intersection(moc_field)

        area_field[i] = moc_field.area_sq_deg

        if area_field[i] == 0:
            nsources_field[i] = 0
        else:
            ## Select all sources in the field
            field_coords = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)

            msk_src = src_catalog.separation(field_coords) <= radius
            src_table_new = src_table[msk_src]

            ## Select sources in the non-overlaping area
            inmoc_table = sources_inmoc(src_table_new, hp, moc_field,
                                        moc_order=moc_order,
                                        ra=rakey, dec=deckey)

            nsources_field[i] = len(inmoc_table)

            # Save sources
            if nsources_field[i] > 0:
                field_table_file = os.path.join(data_folder, groups_folder,
                                                '{}.fits'.format(row['OBS_ID']))
                inmoc_table.keep_columns(['SRCID', raCol, decCol, 'SC_POSERR'])
                inmoc_table.write(field_table_file, overwrite=True)

    colsrc = Table.Column(nsources_field, name='NSRC_XMM')
    colarea = Table.Column(area_field*u.deg**2, name='SKY_AREA')

    if 'EP_TEXP' in obsids_table.colnames:
        obsids_table.add_columns([colarea, colsrc])

    else:
        texp_ep = (3.0*obsids_table['PN_TEXP'] +
                   obsids_table['M1_TEXP'] + obsids_table['M2_TEXP'])/5
        colexp = Table.Column(texp_ep*u.s, name='EP_TEXP')
        obsids_table.add_columns([colexp, colarea, colsrc])

    return obsids_table[colsrc > 0]


def sdss(obsids_table, data_folder, moc_folder, nir_moc=None, data_release=14,
         radius=15*u.arcmin, moc_order=15, overwrite=True):
    """
    Get SDSS data using astroquery.
    For each observation in obsids_table, saves a fits file with
    name 'OBS_ID.fits' in 'data_folder/groups'.

    The function sends a query and selects all sources within 'radius'
    of the RA,DEC of the observation, then it filters the result
    selecting the sources in the corresponding MOC stored in 'moc_folder/mocs'
    (moc_order must be consistent with the order used to calculate the MOC).

    If overwrite is True, always create a new fits file. If False, checks for
    an existing file and uses it to calculate the number of SDSS sources
    in the field. If it doesn't exist, creates the file.

    The function returns obsids_table with an additional column 'NSRC_SDSS'
    with the number of sources in the field.
    """
    # Groups folder
    if nir_moc is None:
        groups_folder = os.path.join(data_folder, 'groups')

    else:
        root, _ = os.path.splitext(os.path.basename(nir_moc))
        survey = root.split('_')[-1]
        groups_folder = os.path.join(data_folder, 'groups_' + survey)

        moc_nirsurvey = MOC()
        read_moc_fits(moc_nirsurvey, nir_moc)

    if not os.path.exists(groups_folder):
        os.makedirs(groups_folder)

    moc_folder = os.path.join(moc_folder, 'mocs')

    nsources_field = np.full((len(obsids_table),), np.nan)
    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())
    photoobj_fields = ['objid', 'mode', 'ra', 'dec', 'raErr', 'decErr']

    for i, row in enumerate(tqdm(obsids_table,
                                 desc="Making SDSS groups")):
        ## Group file name
        field_table_file = os.path.join(data_folder, groups_folder,
                                        '{}.fits'.format(row['OBS_ID']))
        is_field_table = os.path.exists(field_table_file)

        if overwrite or (not overwrite and not is_field_table):
            ## Select all sources in the field
            field_coords = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)

            src_table = SDSS.query_region(field_coords, radius=radius,
                                          photoobj_fields=photoobj_fields,
                                          data_release=data_release)
            # Filter table
            # In ARCHES, the only filter is selecting primary objects,
            # no filtering in the quality of photometry (clean).
            src_table = src_table[src_table['mode'] == 1]

            ## Select sources in the non-overlaping area
            moc_field = MOC()
            read_moc_fits(moc_field, os.path.join(moc_folder,
                                            '{}.moc'.format(row['OBS_ID'])))
            if nir_moc is not None:
                moc_field = moc_nirsurvey.intersection(moc_field)

            inmoc_table = sources_inmoc(src_table, hp, moc_field,
                                        moc_order=moc_order,
                                        ra='ra', dec='dec', units=u.deg)
            ## Save sources
            inmoc_table.remove_columns(['mode'])
            inmoc_table.meta['description'] = 'SDSS'
            inmoc_table.write(field_table_file, overwrite=True)

        else:
            inmoc_table = Table.read(field_table_file)

        nsources_field[i] = len(inmoc_table)

    colsrc = Table.Column(nsources_field, name='NSRC_SDSS')
    obsids_table.add_column(colsrc)

    return obsids_table


def pstarrs(obsids_table, data_folder, moc_folder, nir_moc=None,
            radius=15*u.arcmin, moc_order=16, overwrite=True):
    """
    Get Pan-STARRS data using astroquery and Vizier.
    For each observation in obsids_table, saves a fits file with
    name 'OBS_ID.fits' in 'data_folder/groups'.

    The function sends a Vizier query and selects all sources within 'radius'
    arcmin of the RA,DEC of the observation, then it filters the result
    selecting the sources in the corresponding MOC stored in 'moc_folder/mocs'
    (moc_order must be consistent with the order used to calculate the MOC).

    If overwrite is True, always create a new fits file. If False, checks for
    an existing file and uses it to calculate the number of Pan-STARRS sources
    in the field. If it doesn't exist, creates the file.

    The function returns obsids_table with an additional column 'NSRC_PS' with
    the number of sources in the field.
    """

    # Groups folder
    if nir_moc is None:
        groups_folder = os.path.join(data_folder, 'groups')
    else:
        root, _ = os.path.splitext(os.path.basename(nir_moc))
        survey = root.split('_')[-1]
        groups_folder = os.path.join(data_folder, 'groups_' + survey)

        moc_nirsurvey = MOC()
        read_moc_fits(moc_nirsurvey, nir_moc)

    if not os.path.exists(groups_folder):
        os.makedirs(groups_folder)

    moc_folder = os.path.join(moc_folder, 'mocs')

    nsources_field = np.full((len(obsids_table),), np.nan)
    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())

    v = Vizier(columns=['objID', 'RAJ2000', 'DEJ2000',
                        'e_RAJ2000', 'e_DEJ2000', 'Nd', 'Qual'],
               column_filters={"Nd":">1"}, row_limit=np.inf, timeout=6000)

    for i, row in enumerate(tqdm(obsids_table,
                                 desc="Making Pan-STARRS groups")):
        ## Group file name
        field_table_file = os.path.join(data_folder, groups_folder,
                                        '{}.fits'.format(row['OBS_ID']))
        is_field_table = os.path.exists(field_table_file)

        if overwrite or (not overwrite and not is_field_table):
            ## Select all sources in the field
            field_coords = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)

            vrsp = v.query_region_async(field_coords, radius=radius,
                                        catalog='II/349', return_type='asu-tsv')

            # Fix bug in the vizier response
            # (returns the objID as a short int and fails to load
            # properly as an astropy table)
            with open('/tmp/tmp.tab', 'wb') as tmpfile:
                tmpfile.write(vrsp.content)

            src_table = Table.read('/tmp/tmp.tab', format='ascii.tab')
            src_table = src_table[2:]

            objid = np.array(src_table['objID']).astype(np.int64)
            ra = np.array(src_table['RAJ2000']).astype(np.float) * u.deg
            dec = np.array(src_table['DEJ2000']).astype(np.float) * u.deg

            err_ra = np.array(src_table['e_RAJ2000'])
            err_ra[err_ra == '            '] = '-1'
            err_ra = err_ra.astype(np.float) * u.arcsec
            err_ra[err_ra == -1] = np.nan

            err_dec = np.array(src_table['e_DEJ2000'])
            err_dec[err_dec == '            '] = '-1'
            err_dec = err_dec.astype(np.float) * u.arcsec
            err_dec[err_dec == -1] = np.nan

            flag = np.array(src_table['Qual']).astype(np.int32)

            src_table = Table([objid, ra, dec, err_ra, err_dec, flag],
                              names=['objID', 'RAJ2000', 'DEJ2000',
                                     'e_RAJ2000', 'e_DEJ2000', 'Qual'])
            # Filter table
            msk_good = (src_table['Qual'] & 16) != 0
            src_table_new = src_table[msk_good]

            ## Select sources in the non-overlaping area
            moc_field = MOC()
            read_moc_fits(moc_field, os.path.join(moc_folder,
                                            '{}.moc'.format(row['OBS_ID'])))

            if nir_moc is not None:
                moc_field = moc_nirsurvey.intersection(moc_field)

            inmoc_table = sources_inmoc(src_table_new, hp, moc_field,
                                        moc_order=moc_order,
                                        ra='RAJ2000', dec='DEJ2000')
            ## Save sources
            inmoc_table.remove_columns(['Qual'])
            inmoc_table.meta['description'] = 'Pan-STARRS'
            inmoc_table.write(field_table_file, overwrite=True)

        else:
            inmoc_table = Table.read(field_table_file)

        nsources_field[i] = len(inmoc_table)

    colsrc = Table.Column(nsources_field, name='NSRC_PS')
    obsids_table.add_column(colsrc)

    return obsids_table


def wise(obsids_table, data_folder, moc_folder, nir_moc=None, opt_moc=None,
         radius=15*u.arcmin, moc_order=16, overwrite=True):
    """
    Get All-WISE data using astroquery and Vizier
    For each observation in obsids_table, saves a fits file with
    name 'OBS_ID.fits' in 'data_folder/groups'.

    The function sends a Vizier query and selects all sources within 'radius'
    arcmin of the RA,DEC of the observation, then it filters the result
    selecting the sources in the corresponding MOC stored in 'moc_folder/mocs'
    (moc_order must be consistent with the order used to calculate the moc).

    If overwrite is True, always create a new fits file. If False, checks for
    an existing file and uses it to calculate the number of WISE sources
    in the field. If it doesn't exist, creates the file.

    The function returns obsids_table with an additional column 'NSRC_WS'
    with the number of sources in the field.
    """
    # Groups folder
    if nir_moc is None:
        groups_folder = os.path.join(data_folder, 'groups')
    else:
        root, _ = os.path.splitext(os.path.basename(nir_moc))
        survey = root.split('_')[-1]
        groups_folder = os.path.join(data_folder, 'groups_' + survey)

        moc_nirsurvey = MOC()
        read_moc_fits(moc_nirsurvey, nir_moc)

    if not os.path.exists(groups_folder):
        os.makedirs(groups_folder)

    moc_folder = os.path.join(moc_folder, 'mocs')

    if opt_moc is not None:
        moc_optsurvey = MOC()
        read_moc_fits(moc_optsurvey, opt_moc)

    nsources_field = np.full((len(obsids_table),), np.nan)
    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())

    v = Vizier(columns=['ID', 'RAJ2000', 'DEJ2000', 'eeMaj', 'eeMin', 'eePA'],
               row_limit=np.inf, timeout=6000)

    for i, row in enumerate(tqdm(obsids_table, desc="Making WISE groups")):
        ## Group file name
        field_table_file = os.path.join(data_folder, groups_folder,
                                        '{}.fits'.format(row['OBS_ID']))
        is_field_table = os.path.exists(field_table_file)

        if overwrite or (not overwrite and not is_field_table):
            ## Select all sources in the field
            field_coords = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)

            vrsp = v.query_region(field_coords, radius=radius,
                                  catalog='II/328/allwise')

            ## Select sources in the non-overlaping area
            moc_field = MOC()
            read_moc_fits(moc_field, os.path.join(moc_folder,
                                            '{}.moc'.format(row['OBS_ID'])))
            if opt_moc is not None:
                moc_field = moc_optsurvey.intersection(moc_field)

            if nir_moc is not None:
                moc_field = moc_nirsurvey.intersection(moc_field)

            inmoc_table = sources_inmoc(vrsp[0], hp, moc_field,
                                        moc_order=moc_order,
                                        ra='RAJ2000', dec='DEJ2000')
            ## Save sources
            field_table_file = os.path.join(data_folder, groups_folder,
                                            '{}.fits'.format(row['OBS_ID']))

            inmoc_table.meta['description'] = 'AllWISE'
            inmoc_table.write(field_table_file, overwrite=True)

        else:
            inmoc_table = Table.read(field_table_file)

        nsources_field[i] = len(inmoc_table)

    colsrc = Table.Column(nsources_field, name='NSRC_WS')
    obsids_table.add_column(colsrc)

    return obsids_table


def tmass(obsids_table, data_folder, moc_folder, opt_moc=None,
          radius=15*u.arcmin, moc_order=16, overwrite=True):
    """
    Get 2MASS data using astroquery and Vizier
    For each observation in obsids_table, saves a fits file with
    name 'OBS_ID.fits' in 'data_folder/groups'.

    The function sends a Vizier query and selects all sources within 'radius'
    arcmin of the RA,DEC of the observation, then it filters the result
    selecting the sources in the corresponding MOC stored in 'moc_folder/mocs'
    (moc_order must be consistent with the order used to calculate the moc).

    If overwrite is True, always create a new fits file. If False, checks for
    an existing file and uses it to calculate the number of WISE sources
    in the field. If it doesn't exist, creates the file.

    The function returns obsids_table with an additional column 'NSRC_2M'
    with the number of sources in the field.
    """
    # Groups folder
    groups_folder = os.path.join(data_folder, 'groups')
    if not os.path.exists(groups_folder):
        os.makedirs(groups_folder)

    moc_folder = os.path.join(moc_folder, 'mocs')

    if opt_moc is not None:
        moc_optsurvey = MOC()
        read_moc_fits(moc_optsurvey, opt_moc)

    nsources_field = np.full((len(obsids_table),), np.nan)
    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())

    v = Vizier(columns=['Cntr', 'RAJ2000', 'DEJ2000',
                        'errMaj', 'errMin', 'errPA', 'Qflg'],
               row_limit=np.inf, timeout=6000)

    for i, row in enumerate(tqdm(obsids_table, desc="Making 2MASS groups")):
        ## Group file name
        field_table_file = os.path.join(data_folder, groups_folder,
                                        '{}.fits'.format(row['OBS_ID']))
        is_field_table = os.path.exists(field_table_file)

        if overwrite or (not overwrite and not is_field_table):
            ## Select all sources in the field
            field_coords = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)

            vrsp = v.query_region_async(field_coords, radius=radius,
                                        catalog='II/246/out',
                                        return_type='asu-tsv')

            # Fix bug in the vizier response
            # (returns the id as a short int and fails to load
            # properly as an astropy table)
            with open('/tmp/tmp.tab', 'wb') as tmpfile:
                tmpfile.write(vrsp.content)

            src_table = Table.read('/tmp/tmp.tab', format='ascii.tab')
            src_table = src_table[2:]

            objid = np.array(src_table['Cntr']).astype(np.int64)
            ra = np.array(src_table['RAJ2000']).astype(np.float)*u.deg
            dec = np.array(src_table['DEJ2000']).astype(np.float)*u.deg

            errMaj = np.array(src_table['errMaj']).astype(np.float)*u.arcsec
#            err_ra[err_ra == '            '] = '-1'
#            err_ra = err_ra.astype(np.float) * u.arcsec
#            err_ra[err_ra == -1] = np.nan

            errMin = np.array(src_table['errMin']).astype(np.float)*u.arcsec
#            err_dec[err_dec == '            '] = '-1'
#            err_dec = err_dec.astype(np.float) * u.arcsec
#            err_dec[err_dec == -1] = np.nan

            errPA = np.array(src_table['errPA']).astype(np.float)*u.deg
            flag = np.array(src_table['Qflg'])

            src_table = Table([objid, ra, dec, errMaj, errMin, errPA, flag],
                              names=['objID', 'RAJ2000', 'DEJ2000',
                                     'errMaj', 'errMin', 'errPA', 'Qflg'])
            # Filter table
            # Sources detected with SNR>=5 in J, H or K
            flgJ = [f[0] in ['A', 'B', 'C'] for f in src_table['Qflg']]
            flgH = [f[1] in ['A', 'B', 'C'] for f in src_table['Qflg']]
            flgK = [f[2] in ['A', 'B', 'C'] for f in src_table['Qflg']]
            msk_good = np.logical_and(flgJ, np.logical_and(flgH, flgK))
            src_table_new = src_table[msk_good]

            ## Select sources in the non-overlaping area
            moc_field = MOC()
            read_moc_fits(moc_field, os.path.join(moc_folder,
                                            '{}.moc'.format(row['OBS_ID'])))
            if opt_moc is not None:
                moc_field = moc_optsurvey.intersection(moc_field)

            inmoc_table = sources_inmoc(src_table_new, hp, moc_field,
                                        moc_order=moc_order,
                                        ra='RAJ2000', dec='DEJ2000')
            ## Save sources
            field_table_file = os.path.join(data_folder, groups_folder,
                                            '{}.fits'.format(row['OBS_ID']))

            inmoc_table.meta['description'] = '2MASS'
            inmoc_table.remove_column('Qflg')
            inmoc_table.write(field_table_file, overwrite=True)

        else:
            inmoc_table = Table.read(field_table_file)

        nsources_field[i] = len(inmoc_table)

    colsrc = Table.Column(nsources_field, name='NSRC_2M')
    obsids_table.add_column(colsrc)

    return obsids_table


def ukidss(obsids_table, data_folder, moc_folder, opt_moc=None,
           radius=15*u.arcmin, moc_order=16, overwrite=True):
    """
    Get UKIDSS-LAS data using astroquery and the UKIDSS database
    For each observation in obsids_table, saves a fits file with
    name 'OBS_ID.fits' in 'data_folder/groups'.

    The function sends a query and selects all sources within 'radius'
    arcmin of the RA,DEC of the observation, then it filters the result
    selecting the sources in the corresponding MOC stored in 'moc_folder/mocs'
    (moc_order must be consistent with the order used to calculate the moc).

    If overwrite is True, always create a new fits file. If False, checks for
    an existing file and uses it to calculate the number of UKIDSS sources
    in the field. If it doesn't exist, creates the file.

    The function returns obsids_table with an additional column 'NSRC_UK'
    with the number of sources in the field.
    """
    # Groups folder
    groups_folder = os.path.join(data_folder, 'groups')
    if not os.path.exists(groups_folder):
        os.makedirs(groups_folder)

    moc_folder = os.path.join(moc_folder, 'mocs')

    if opt_moc is not None:
        moc_optsurvey = MOC()
        read_moc_fits(moc_optsurvey, opt_moc)

    nsources_field = np.full((len(obsids_table),), np.nan)
    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())

    v = Ukidss()
    columns = 'sourceID, RA, Dec, sigRA, sigDec'
    constraint = '(j_1ppErrBits | hppErrBits | kppErrBits) < 65536'

    for i, row in enumerate(tqdm(obsids_table, desc="Making UKIDSS groups")):
        ## Group file name
        field_table_file = os.path.join(data_folder, groups_folder,
                                        '{}.fits'.format(row['OBS_ID']))
        is_field_table = os.path.exists(field_table_file)

        if overwrite or (not overwrite and not is_field_table):
            ## Select all sources in the field
            field_coords = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)

            vrsp = v.query_region(field_coords, radius=radius,
                                  database='UKIDSSDR10PLUS', programme_id='LAS',
                                  select=columns, where=constraint)

            # Change bad coords errors and add units
            msk = vrsp['sigRA'] < 0
            err = np.array(vrsp['sigRA'])
            err[msk] = 0.1/3600
            err = err*u.deg
            vrsp.replace_column('sigRA', Table.Column(err.to(u.arcsec)))

            msk = vrsp['sigDec'] < 0
            err = np.array(vrsp['sigDec'])
            err[msk] = 0.1/3600
            err = err*u.deg
            vrsp.replace_column('sigDec', Table.Column(err.to(u.arcsec)))

            vrsp['RA'] = vrsp['RA']*u.deg
            vrsp['Dec'] = vrsp['Dec']*u.deg
            vrsp.rename_column('sourceID', 'objID') # for consistency with 2MASS
            vrsp.remove_column('distance')

            ## Select sources in the non-overlaping area
            moc_field = MOC()
            read_moc_fits(moc_field, os.path.join(moc_folder,
                                            '{}.moc'.format(row['OBS_ID'])))
            if opt_moc is not None:
                moc_field = moc_optsurvey.intersection(moc_field)

            inmoc_table = sources_inmoc(vrsp, hp, moc_field,
                                        moc_order=moc_order,
                                        ra='RA', dec='Dec')
            ## Save sources
            field_table_file = os.path.join(data_folder, groups_folder,
                                            '{}.fits'.format(row['OBS_ID']))

            inmoc_table.meta['description'] = 'UKIDSS'
            inmoc_table.write(field_table_file, overwrite=True)

        else:
            inmoc_table = Table.read(field_table_file)

        nsources_field[i] = len(inmoc_table)

    colsrc = Table.Column(nsources_field, name='NSRC_UK')
    obsids_table.add_column(colsrc)

    return obsids_table


def vista(obsids_table, data_folder, moc_folder, opt_moc=None,
          radius=15*u.arcmin, moc_order=16, overwrite=True):
    """
    Get VISTA-VHS data using astroquery and the UKIDSS database
    For each observation in obsids_table, saves a fits file with
    name 'OBS_ID.fits' in 'data_folder/groups'.

    The function sends a query and selects all sources within 'radius'
    arcmin of the RA,DEC of the observation, then it filters the result
    selecting the sources in the corresponding MOC stored in 'moc_folder/mocs'
    (moc_order must be consistent with the order used to calculate the moc).

    If overwrite is True, always create a new fits file. If False, checks for
    an existing file and uses it to calculate the number of UKIDSS sources
    in the field. If it doesn't exist, creates the file.

    The function returns obsids_table with an additional column 'NSRC_VT'
    with the number of sources in the field.
    """
    # Groups folder
    groups_folder = os.path.join(data_folder, 'groups')
    if not os.path.exists(groups_folder):
        os.makedirs(groups_folder)

    moc_folder = os.path.join(moc_folder, 'mocs')

    if opt_moc is not None:
        moc_optsurvey = MOC()
        read_moc_fits(moc_optsurvey, opt_moc)

    nsources_field = np.full((len(obsids_table),), np.nan)
    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())

    v = Vista()
    columns = 'sourceID, RA, Dec'
    constraint = '(jppErrBits | hppErrBits | ksppErrBits) < 65536'

    for i, row in enumerate(tqdm(obsids_table, desc="Making VISTA groups")):
        ## Group file name
        field_table_file = os.path.join(data_folder, groups_folder,
                                        '{}.fits'.format(row['OBS_ID']))
        is_field_table = os.path.exists(field_table_file)

        if overwrite or (not overwrite and not is_field_table):
            ## Select all sources in the field
            field_coords = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)

            vrsp = v.query_region(field_coords, radius=radius,
                                  database='VHSDR4', programme_id='VHS',
                                  select=columns, where=constraint)

            # Add error column and units
            err = np.full((len(vrsp),), 0.1)*u.arcsec
            vrsp.add_column(Table.Column(err, name='RADECERR'))

            vrsp['RA'] = vrsp['RA']*u.deg
            vrsp['Dec'] = vrsp['Dec']*u.deg
            vrsp.rename_column('sourceID', 'objID') # for consistency with 2MASS
            vrsp.remove_column('distance')

            ## Select sources in the non-overlaping area
            moc_field = MOC()
            read_moc_fits(moc_field, os.path.join(moc_folder,
                                            '{}.moc'.format(row['OBS_ID'])))
            if opt_moc is not None:
                moc_field = moc_optsurvey.intersection(moc_field)

            inmoc_table = sources_inmoc(vrsp, hp, moc_field,
                                        moc_order=moc_order,
                                        ra='RA', dec='Dec')
            ## Save sources
            field_table_file = os.path.join(data_folder, groups_folder,
                                            '{}.fits'.format(row['OBS_ID']))

            inmoc_table.meta['description'] = 'VISTA'
            inmoc_table.write(field_table_file, overwrite=True)

        else:
            inmoc_table = Table.read(field_table_file)

        nsources_field[i] = len(inmoc_table)

    colsrc = Table.Column(nsources_field, name='NSRC_VT')
    obsids_table.add_column(colsrc)

    return obsids_table
