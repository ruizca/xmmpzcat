# -*- coding: utf-8 -*-

import numpy as np
import os
from tqdm import trange, tqdm

from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

from pymoc import MOC
from pymoc.util.catalog import catalog_to_moc
from pymoc.io.fits import read_moc_fits


def make_mocs(obsids_table, data_folder, moc_order=16, 
              radius=15*u.arcmin, remove_stars=False) :

    # MOC folder
    moc_folder = os.path.join(data_folder, 'mocs')
    if not os.path.exists(moc_folder) :
        os.makedirs(moc_folder)
    
    # Calculate EPIC exposure time and sort the obsids
    texp_m1 = obsids_table['M1_TEXP']
    texp_m2 = obsids_table['M2_TEXP']
    texp_pn = obsids_table['PN_TEXP']
    texp_ep = (3*texp_pn + texp_m1 + texp_m2)/5    
    obsids_table.add_column(texp_ep, name='EP_TEXP')
    obsids_table.sort('EP_TEXP')
    
    obsids = obsids_table['OBS_ID']    
    fields_catalog = SkyCoord(ra=obsids_table['RA']*u.degree, 
                              dec=obsids_table['DEC']*u.degree)     

    v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag'], 
               column_filters={"Vmag":"<15"}, row_limit=np.inf, timeout=6000)
    
    for i in trange(len(obsids), desc="Making MOCs") :
        moc_filename = os.path.join(moc_folder, '{}.moc'.format(obsids[i]))

        # Finding overlapping fields
        field_coords = fields_catalog[i]
        new_catalog = fields_catalog[i+1:]
        d2d = field_coords.separation(new_catalog) 
        catalogmsk1 = d2d <= 2*radius
        catalogmsk2 = d2d > 0*u.arcmin
        catalogmsk = np.logical_and(catalogmsk1, catalogmsk2)
        idxcatalog = np.where(catalogmsk)[0]
        
        # Making MOC
        if len(idxcatalog) == 0 :
            moc_field = catalog_to_moc(field_coords, radius, moc_order)
            
        else :
            moc_originalField = catalog_to_moc(field_coords, radius, moc_order)
            moc_bad =  catalog_to_moc(new_catalog[idxcatalog], radius, moc_order)
        
            moc_field = moc_originalField - moc_bad
        
        if remove_stars :
            # Mask bright stars (V<15) from the field 
            # (HST Guide Star Catalog, Version 2.3.2 (GSC2.3))
            vrsp = v.query_region(field_coords, radius=(radius + 1.5*u.arcmin), 
                                  catalog='I/305')
            
            if len(vrsp) > 0 :
                stars = vrsp[0]
                stars_coords = SkyCoord(ra=stars['RAJ2000'], dec=stars['DEJ2000'])
                stars_radius = (16 - stars['Vmag'])*6*u.arcsec # Aird+2015
    
                moc_stars = MOC()
                for coord, r in zip(stars_coords, stars_radius) :
                    moc_stars += catalog_to_moc(coord, r, moc_order)
                    
                moc_field = moc_field - moc_stars

        moc_field.write(moc_filename, filetype="FITS", overwrite=True)


def make_binmocs(obsids_table, data_folder, bincol='BIN_ID') :
    
    bins = np.unique(obsids_table[bincol])
    bins = bins[~np.isnan(bins)]
    
    mocgroups_folder = os.path.join(data_folder, 'mocs')
    binmocs_folder = os.path.join(data_folder, 'binmocs')
    if not os.path.exists(binmocs_folder) :
        os.makedirs(binmocs_folder)

    for binid in tqdm(bins, desc='Making bin MOCs') :
        msk_bin = obsids_table[bincol] == binid
        bin_table = obsids_table[msk_bin]
        
        binmoc_filename = os.path.join(
                            binmocs_folder, 
                            'bin{}.moc'.format(str(int(binid)).zfill(3)))
    
        moc_bin = MOC()
        for field in bin_table['OBS_ID'] :
            moc_field_file = os.path.join(mocgroups_folder, 
                                          '{}.moc'.format(field))
            moc_field = MOC()
            read_moc_fits(moc_field, moc_field_file)
            
            moc_bin = moc_bin + moc_field
            
        moc_bin.write(binmoc_filename, filetype="FITS", overwrite=True)