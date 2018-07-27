# -*- coding: utf-8 -*-
import os

from astropy import wcs
from astropy.coordinates import SkyCoord, ICRS
from astropy import units as u
from astropy_healpix import HEALPix
from regions import EllipseSkyRegion, CircleSkyRegion
from astroquery.vizier import Vizier
from astroquery.irsa import Irsa
from pymoc import MOC
from pymoc.util.catalog import catalog_to_moc
from pymoc.io.fits import read_moc_fits
from tqdm import trange, tqdm
import numpy as np


def sort_exptime(table):
    # Calculate EPIC exposure time and sort the obsids
    texp_m1 = table['M1_TEXP']
    texp_m2 = table['M2_TEXP']
    texp_pn = table['PN_TEXP']
    table['EP_TEXP'] = (3*texp_pn + texp_m1 + texp_m2)/5
    table.sort('EP_TEXP')


def overlap_fields(ncat, fields, radius):
    """
    Find fields that overlap within radius radius with the ncat field .
    """
    new_catalog = fields[ncat + 1:]
    d2d = fields[ncat].separation(new_catalog)
    catalogmsk = np.logical_and(d2d <= 2*radius, d2d > 0*u.arcmin)

    return new_catalog[np.where(catalogmsk)]


def basic_moc(field, overlap, radius, moc_order=15):
    """
    Circular MOC for field, removing other overlapping fields if they exist.
    """
    if len(overlap):
        moc_originalField = catalog_to_moc(field, radius, moc_order)
        moc_bad = catalog_to_moc(overlap, radius, moc_order)
        moc_field = moc_originalField - moc_bad

    else:
        moc_field = catalog_to_moc(field, radius, moc_order)

    return moc_field


def galaxies_moc(field, moc_field, radius, moc_order=15):
    """
    MOC with the intersection of field with the galaxies
    included in the 2MASS Large Galaxy Atlas
    """
    galaxies = Irsa.query_region(field, catalog="lga_v2",
                                 spatial="Cone", radius=2*u.deg)
    moc_galaxies = MOC()
    if len(galaxies):
        w = obsid_wcs(field)
        field_reg = CircleSkyRegion(center=field, radius=radius)

        moc_galaxies = MOC()
        for g in galaxies:
            gcoords = SkyCoord(ra=g['ra'], dec=g['dec'], unit=u.deg)
            errMaj = 1.5*2*g['r_ext']*u.arcsec
            galaxy_reg = EllipseSkyRegion(center=gcoords,
                            width=errMaj, height=errMaj*g['sup_ba'],
                            angle=(90 + g['sup_pa'])*u.deg)

            region = field_reg.intersection(galaxy_reg)
            moc_galaxies += reg2moc(region, moc_field, w, moc_order)

    return moc_galaxies


def stars_moc(field, radius, moc_order=15):
    """
    MOC with the field area cover by bright stars (V<15).
    (HST Guide Star Catalog, Version 2.3.2 (GSC2.3))
    """
    v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag'],
               column_filters={"Vmag":"<15"},
               row_limit=np.inf, timeout=6000)

    vrsp = v.query_region(field, radius=(radius + 1.5*u.arcmin), catalog='I/305')

    moc_stars = MOC()
    if len(vrsp) > 0:
        stars = vrsp[0]
        stars_coords = SkyCoord(ra=stars['RAJ2000'], dec=stars['DEJ2000'])
        stars_radius = (16 - stars['Vmag'])*6*u.arcsec # Aird+2015

        for coord, r in zip(stars_coords, stars_radius):
            moc_stars += catalog_to_moc(coord, r, moc_order)

    return moc_stars


def obsid_wcs(coords):
    """
    WCS object with a gnomonic projection centered in coords.
    """
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)

    # Set up an "gnomonic" projection (as in XMM EPIC images)
    # Vector properties may be set with Python lists, or Numpy arrays
    w.wcs.crpix = [2.98436767602103E+02, 2.98436767602103E+02]
    w.wcs.cdelt = np.array([-1.20833333333333E-03, 1.20833333333333E-03])
    w.wcs.crval = [coords.ra.deg, coords.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 0.0)])

    return w


def reg2moc(region, moc_field, wcs, moc_order=15):
    """
    Transform a the intersection of moc_field with a Region object into a MOC.
    """
    list_cells = np.array([c for c in moc_field.flattened()])

    hp = HEALPix(nside=2**moc_order, order='nested', frame=ICRS())
    cells_skycoords = hp.healpix_to_skycoord(list_cells)

    mask = region.contains(cells_skycoords, wcs)
    region_moc = MOC(moc_order, list_cells[mask])

    return region_moc


def make_mocs(obsids_table, data_folder, moc_order=15, radius=15*u.arcmin,
              remove_stars=False, remove_large_galaxies=False):
    """
    Create non-overlapping mocs for the observations included in obsids_table.
    """
    moc_folder = os.path.join(data_folder, 'mocs')
    if not os.path.exists(moc_folder):
        os.makedirs(moc_folder)

    sort_exptime(obsids_table)

    obsids = obsids_table['OBS_ID']
    fields = SkyCoord(ra=obsids_table['RA'], dec=obsids_table['DEC'], unit=u.deg)

    for i in trange(len(obsids), desc="Making MOCs"):
        moc_filename = os.path.join(moc_folder, '{}.moc'.format(obsids[i]))

        fields_overlap = overlap_fields(i, fields, radius)

        moc_field = basic_moc(fields[i], fields_overlap, radius, moc_order)

        if remove_large_galaxies and moc_field.area > 0:
            moc_field -= galaxies_moc(fields[i], moc_field, radius, moc_order)

        if remove_stars and moc_field.area > 0:
            moc_field -= stars_moc(fields[i], radius, moc_order)

        moc_field.normalize()
        moc_field.write(moc_filename, filetype="FITS", overwrite=True)


def make_binmocs(obsids_table, data_folder, bincol='BIN_ID'):

    bins = np.unique(obsids_table[bincol])
    bins = bins[~np.isnan(bins)]

    mocgroups_folder = os.path.join(data_folder, 'mocs')
    binmocs_folder = os.path.join(data_folder, 'binmocs')
    if not os.path.exists(binmocs_folder):
        os.makedirs(binmocs_folder)

    for binid in tqdm(bins, desc='Making bin MOCs'):
        msk_bin = obsids_table[bincol] == binid
        bin_table = obsids_table[msk_bin]

        binmoc_filename = os.path.join(binmocs_folder,
                            'bin{}.moc'.format(str(int(binid)).zfill(3)))

        moc_bin = MOC()
        for field in bin_table['OBS_ID']:
            moc_field_file = os.path.join(mocgroups_folder,
                                          '{}.moc'.format(field))
            moc_field = MOC()
            read_moc_fits(moc_field, moc_field_file)
            moc_bin += moc_field

        moc_bin.write(binmoc_filename, filetype="FITS", overwrite=True)


