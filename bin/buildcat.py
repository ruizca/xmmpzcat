#!/usr/bin/env python
"""
Created on Sat Feb  3 14:42:22 2018

Build XMMPZCAT from scratch

@author: A. Ruiz
"""
import os
from astropy.table import Table


def main():
    """
    Build XMMPZCAT from scratch.
    """
    make_xmatch = True
    get_photometry = False
    calc_photoz = False
    id_stars = False
    opt_survey = 'sdss'
    photoz_folder = '../photoz/'

    if make_xmatch:
        import probacats as pc

        xmatchcat_tmass = pc.make_cat(opt_survey, '2MASS', poscorr=False, 
                                      make_mocs=False, getXdata=False, 
                                      getOPTdata=False, getWSdata=False, 
                                      getNIRdata=False, define_bins=False, 
                                      make_bins=False, make_xmatch=True)

        xmatchcat_ukidss = pc.make_cat(opt_survey, 'UKIDSS', poscorr=False, 
                                      make_mocs=False, getXdata=True, 
                                      getOPTdata=True, getWSdata=True, 
                                      getNIRdata=True, define_bins=True, 
                                      make_bins=True, make_xmatch=True)

        xmatchcat_vista = pc.make_cat(opt_survey, 'VISTA', poscorr=False, 
                                      make_mocs=False, getXdata=True, 
                                      getOPTdata=True, getWSdata=True, 
                                      getNIRdata=True, define_bins=True, 
                                      make_bins=True, make_xmatch=True)

        xmatchcat_final = pc.merge_cat(xmatchcat_tmass,
                                       xmatchcat_ukidss,
                                       xmatchcat_vista)
        xmatchcat_final = pc.clean_cat(xmatchcat_final, probalimit=0.68)
        xmatchcat_final.write('../xmatch/xmatchcat.fits', overwrite=True)

    else:
        xmatchcat_final = Table.read('../xmatch/xmatchcat.fits', memmap=True)


    photometry_file = '../xmatch/xmatchcat_photometry.fits'
    if get_photometry:
        import photometry

        xmatchcat_photom = photometry.add_data(xmatchcat_final, getPSphoto=True,
                                               getWSphoto=True, getUKphoto=True,
                                               getVTphoto=True, getTMphoto=True,
                                               extinction=True)
        xmatchcat_photom.write(photometry_file, overwrite=True)

    else:
        xmatchcat_photom = Table.read(photometry_file, memmap=True)


    photoz_file = os.path.join(photoz_folder, 'xmatchcat_photoz.fits')
    if calc_photoz:
        import photoz

        xmatchcat_photoz = photoz.calc(xmatchcat_photom, photoz_folder)
        xmatchcat_photoz.write(photoz_file, overwrite=True)

    else:
        xmatchcat_photoz = Table.read(photoz_file, memmap=True)


    stars_file = '../xmatch/xmatchcat_stars.fits'
    if id_stars:
        import idstars

        xmatchcat_stars = idstars.find(xmatchcat_photom)
        xmatchcat_stars.write(stars_file, overwrite=True)

    else:
        xmatchcat_stars = Table.read(stars_file, memmap=True)


    finalcat_file = '../xmatch/xmatchcat_final.fits'
    finalcat = pc.final_cat(xmatchcat_stars, xmatchcat_photoz)
    finalcat.write(finalcat_file, overwrite=True)


if __name__ == '__main__':
    main()
