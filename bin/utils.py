# -*- coding: utf-8 -*-

import os
import gzip
import tarfile
import numpy as np

from urllib.request import urlopen
from urllib.error import HTTPError, URLError

from astropy.coordinates import SkyCoord

from pymoc import MOC
from pymoc.io.fits import read_moc_fits

def downloadFile(url, dest_folder, filename=None) :
    # Open the url
    try:
        f = urlopen(url)

        # Open our local file for writing        
        if filename is None : 
            print("Downloading " + url)
            dest_file = os.path.join(dest_folder, os.path.basename(url))
        else :
            dest_file = os.path.join(dest_folder, filename)

        with open(dest_file, "wb") as local_file:
            local_file.write(f.read())

    #handle errors
    except HTTPError as e:
        print("HTTP Error:", e.code, url)
    except URLError as e:
        print("URL Error:", e.reason, url)

def untarFile(input_file, dest_folder, remove=True) :
    f = tarfile.open(input_file)
    f.extractall(path=dest_folder)
    
def gunzipFile(input_file, output_file, remove=True) :
    """
    gunzip input_file, save to output_file. If remove is True, input_file is
    deleted.
    """
    inF = gzip.open(input_file, 'rb')
    outF = open(output_file, 'wb')
    outF.write( inF.read() )
    inF.close()
    outF.close()
    
    if remove :
        os.remove(input_file)

def get_moc(url, survey, folder) :
    """
    Get the moc of the area covered by survey from url and store it in folder.
    """
    if survey is 'UKIDSS' :
        filenameJ = 'las-J1-DR10.fits'
        filenameH = 'las-H-DR10.fits'
        filenameK = 'las-K-DR10.fits'

    elif survey is 'VISTA' :
        filenameJ = 'vhs-J-dr4.fits'
        filenameH = 'vhs-H-dr4.fits'
        filenameK = 'vhs-Ks-dr4.fits'
        
    elif survey is '2MASS' :
        return None
    
    else :
        raise ValueError('Invalid near-infrared survey!')

    filename = os.path.join(folder, 'moc_{}.fits'.format(survey.lower()))

    if not os.path.isfile(filename) :
        # J moc
        moc_J = MOC()  
        downloadFile(os.path.join(url, filenameJ), folder)
        read_moc_fits(moc_J, os.path.join(folder, filenameJ))
    
        # H moc
        moc_H = MOC()
        downloadFile(os.path.join(url, filenameH), folder)
        read_moc_fits(moc_H, os.path.join(folder, filenameH))
        
        # K moc
        moc_K = MOC()
        downloadFile(os.path.join(url, filenameK), folder)
        read_moc_fits(moc_K, os.path.join(folder, filenameK))
    
        moc_JH = moc_J.intersection(moc_H)
        moc_JHK = moc_JH.intersection(moc_K)
        moc_JHK.write(filename, filetype="FITS", overwrite=True)        

    return filename
    
    
def sources_inmoc(sources, hp, moc, moc_order=16, ra='RA', dec='Dec') :
    """
    Find those sources in the astropy table sources included in moc.
    """

    cells = hp.skycoord_to_healpix(SkyCoord(ra=sources[ra], 
                                            dec=sources[dec]))

    cells_msk = np.full(len(cells), np.nan)    
    for j, cell in enumerate(cells) :
        cells_msk[j] = moc.contains(moc_order, cell)
    
    sources_inmoc = sources[np.where(cells_msk)]
    
    return sources_inmoc
    