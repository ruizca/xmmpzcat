# -*- coding: utf-8 -*-
"""
Python parser for the xmatch web service, a cross-matching tool of 
astronomical catalogues. An implementation of the method described 
in Pineau et al. 2017

@author: A. Ruiz
"""

import os
import subprocess as sp

from astropy.table import Table
from astropy import units as u

def login() :
    while True :
        sh_return = sp.check_output('./xmatch.sh i', shell=True)
        
        if sh_return.split()[2] == b'successfully' :
            break
    
def logout() :
    sp.call('./xmatch.sh q', shell=True)
    
def rmfile(filename) :
    sp.call('./xmatch.sh r {}'.format(filename), shell=True)
    
def addfile(filename) :
    while True :
        sh_return = sp.check_output('./xmatch.sh p {}'.format(filename), 
                                    shell=True)
        
        if sh_return.split()[2] == b'successfully' :
            break

def getfile(filename) :
    sp.call('./xmatch.sh g {}'.format(filename), shell=True)
   

def makeXmsFile(catalogues, prefixes, completeness) :
    
    plabels = [s[0] for s in prefixes]    
    joins = 'I'
    proba_letters = ''.join('{},'.format(l) for l in plabels[:-1])

    xms_cmd = 'gset proba_letters={}{}\n'.format(proba_letters, plabels[-1])
    for i, cat in enumerate(catalogues) :            
        cols = cat.colnames
        area = cat.meta['AREA']*u.deg**2
                
        xms_cmd += '\nget FileLoader file=temp_{}.fits\n'.format(prefixes[i])
        xms_cmd += 'set pos ra={} dec={}\n'.format(cols[1], cols[2])
        xms_cmd += 'set poserr type={}'.format(cat.meta['errortype'].upper())
        xms_cmd += ''.join(' param{:d}={}'.format(j+1,p) 
                           for j,p in enumerate(cols[3:]))
        xms_cmd += '\nset cols {}\n'.format(cols[0])
        xms_cmd += 'prefix {}\n'.format(prefixes[i].upper())

        if i > 1 :
            joins += 'I'        
    xms_cmd += '\nxmatch probaN_v1 joins={} '.format(joins)
    xms_cmd += 'completeness={} area={}\n'.format(completeness, area.to(u.rad**2).value)
    xms_cmd += 'save temp_match.fits fits'
    
    fp = open('temp.xms', 'w+')
    fp.write(xms_cmd)
    fp.close()

    
def xmatch(*catalogues, **kwargs) :
    """
    Postional cross-matching of the catalogues in *catalogues. Each 
    catalogue in the list should be an astropy Table with the following 
    structure:
        col1 : source id
        col2 : ra in degrees
        col3 : dec in degrees
        col4 : err_param1
        col5 : err_param2 (optional) 
        col6 : err_param3 (optional)
        
    All tables should have at least two meta keywords, one named "AREA", 
    with the area in squared degrees covered by the catalogue, and one 
    named "errortype", having one of these values: "circle", "ellipse", 
    "rcd_dec_ellipse", "cov_ellipse" or "cor_ellipse".
    
    The number of error parameters depends on errortype: 
      "circle": err_param1 
                (radius of the 1-sigma error circle, in arcsec)
    
      "ellipse": err_param1 
                 (semi-major axis of the 1-sigma error ellipse, in arcsec) 
                 err_param2 
                 (semi-minor axis of the 1-sigma error ellipse, in arcsec)
                 err_param3 
                 (position angle of the 1-sigma error ellipse, in degrees)
    
      "rcd_dec_ellipse": err_param1 
                         (1-sigma error on ra*cos(dec), in arcsec)
                         err_param2 
                         (1-sigma error on dec, in arcsec) 
    
      "cov_ellipse": err_param1 
                     (1-sigma error on ra*cos(dec), in arcsec)
                     err_param2 
                     (1-sigma error on dec, in arcsec)
                     err_param3 
                     (covariance of the 1-sigma errors, in arcsec2)
    
      "cor_ellipse" : err_param1 
                      (1-sigma error on ra*cos(dec), in arcsec)
                      err_param2 
                      (1-sigma error on dec, in arcsec)
                      err_param3 
                      (correlation (rho) between the 1-sigma errors)
    
    See Sect. 4.2 of Pineau et al. 2017 for a detailed discussion of 
    the positional errors that can be found in common astronomical 
    catalogues.
    
    
    This function returns an astropy Table with the list of likely 
    associations for the sources in catalogues, the weighted mean 
    positions and errors, the square of the Mahalanobis distance (chi2) 
    and the corresponding probabilities.
    """   

    ncat = len(catalogues)
    
    # Load kwargs
    kwargsdict = {}
    expected_args = ['completeness', 'catalog_prefixes']
    for key in kwargs.keys():
        if key in expected_args:
            kwargsdict[key] = kwargs[key]
        else:
            raise ValueError('Unexpected Argument')

    if 'completeness' in kwargs.keys() :
        completeness = kwargsdict['completeness']
        
        if not (completeness>0 and completeness<1) :
            raise ValueError('completeness must be a value between 0 and 1!')
    else :
        completeness = 0.997300203

    if 'catalog_prefixes' in kwargs.keys() :
        prefixes = kwargsdict['catalog_prefixes']        
        
        if len(prefixes) != ncat :
            raise ValueError('There must be one prefix per catalogue!')
    
    else :
        raise ValueError('Argument catalog_prefixes is mandatory!') 

    makeXmsFile(catalogues, prefixes, completeness)    
    sp.call('./xmatch.sh x temp.xms', shell=True)
    getfile('temp_match.fits')
    xmatch_table = Table.read('temp_match.fits', memmap=True)

    rmfile('temp_match.fits')
    os.remove('temp.xms')
    
    return xmatch_table
    

