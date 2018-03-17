#!/usr/bin/env python
"""
Created on Sat Feb  3 14:42:22 2018

Build XMMPZCAT from scratch

@author: A. Ruiz

"""

import os
from astropy.table import Table

import probacats as pc
import photometry    
import photoz
import idstars

def main():
    """
    Build XMMPZCAT from scratch.
    """
    merge_cats = False
    get_photometry = False
    calc_photoz = True
 
    photoz_folder = '../photoz/'
        
    if merge_cats:        
        xmatchcat_tmass = pc.make_cat('2MASS')
        xmatchcat_ukidss = pc.make_cat('UKIDSS')
        xmatchcat_vista = pc.make_cat('VISTA')
    
        xmatchcat_final = pc.merge_cat(xmatchcat_tmass, 
                                       xmatchcat_ukidss, 
                                       xmatchcat_vista)
        xmatchcat_final = pc.clean_cat(xmatchcat_final, probalimit=0.68)
        xmatchcat_final.write('../xmatch/xmatchcat.fits', overwrite=True)        
    else:
        xmatchcat_final = Table.read('../xmatch/xmatchcat.fits', memmap=True)


    photometry_file = '../xmatch/xmatchcat_photometry.fits'
    if get_photometry:
        xmatchcat_photom = photometry.add_data(xmatchcat_final)
        xmatchcat_photom.write(photometry_file, overwrite=True)    
    else:
        xmatchcat_photom = Table.read(photometry_file, memmap=True)
    
    photoz_file = os.path.join(photoz_folder, 'xmatchcat_photoz.fits')
    if calc_photoz:
        xmatchcat_photoz = photoz.calc(xmatchcat_photom, photoz_folder)
        xmatchcat_photoz.write(photoz_file, overwrite=True)
    else:
        xmatchcat_photoz = Table.read(photoz_file)
    
    idstars.find(xmatchcat_photom)
    #print(kk)
    
if __name__ == '__main__':
    main()
