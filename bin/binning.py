# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u

#import matplotlib
#matplotlib.use("qt5agg")
#import matplotlib.pyplot as plt


def optical(obsids_table, data_folder, nir_survey='2MASS') :
    """
    Group observations in obsids_table in bins of roughly equal density
    of optical sources, according to the Galactic latitude of the observations.
    """
        
    ## Define Galactic regions
    regions_limits = [0, 10, 20, 25, 40, 60, 90] 
    regions = np.array([regions_limits[0:-1], regions_limits[1:]]).T * u.deg

    eq_coords = SkyCoord(ra=obsids_table['RA']*u.deg,
                         dec=obsids_table['DEC']*u.deg, frame='icrs')
    gal_lat = eq_coords.galactic.b
    density = obsids_table['NSRC_PS']/obsids_table['SKY_AREA']

    region_stats = np.full((len(regions), 8), np.nan)    
    msk_outliers = np.array(len(obsids_table)*[False])
    optbins = np.array(len(obsids_table)*['kk'], dtype='object')
    
    for i, reg in enumerate(regions) :
        msk_reg = np.logical_and(np.abs(gal_lat) >= reg[0], 
                                 np.abs(gal_lat) < reg[1])
                                 
        bin_table = obsids_table[msk_reg]
        median = np.median(density[msk_reg])
        smad = 1.4826*np.median(np.abs(density[msk_reg] - median))

        region_stats[i, 0] = median
        region_stats[i, 1] = smad
        region_stats[i, 2] = np.median(np.abs(gal_lat[msk_reg].value))
        region_stats[i, 3] = region_stats[i, 2] - reg[0].value
        region_stats[i, 4] = reg[1].value - region_stats[i, 2]
        region_stats[i, 5] = np.sum(bin_table['SKY_AREA'])
        region_stats[i, 6] = np.sum(bin_table['NSRC_PS'])
        region_stats[i, 7] = len(bin_table)
    
        # Find density outliers outside the Galactic plane   
        if reg[1] > 20 * u.deg :
            msk_outliers = np.logical_or(msk_outliers, np.logical_and(msk_reg, 
                                         density > median + 10*smad))
        
        name_bin = 'bin{:02d}:{:02d}'.format(int(reg[0].value), 
                                                int(reg[1].value))

        optbins[msk_reg] = name_bin
    
    col_outliers = Table.Column(msk_outliers, name='SKY_OUTLIER')
    col_optbins = Table.Column(optbins, name='OPTBIN', dtype='str')
    obsids_table.add_columns([col_outliers, col_optbins])    
        
#    plt.figure()
#    plt.plot(np.abs(gal_lat), density, lw=0, marker='.', ms=2)
#    plt.scatter(np.abs(gal_lat[msk_outliers]), density[msk_outliers], color='gray')
#    plt.errorbar(region_stats[:,2], region_stats[:,0], 
#                 xerr=(region_stats[:, 3], region_stats[:, 4]), 
#                 yerr=10*region_stats[:,1], fmt='ro', capsize=5, zorder=1000)
#    plt.ylim(ymin=0)
#    plt.show()
    
    region_stats[:,3] = region_stats[:,2] - region_stats[:,3]
    region_stats[:,4] = region_stats[:,4] + region_stats[:,2]
    
    optbins_table = Table(region_stats, names=['SKYDEN_MEDIAN', 'SKYDEN_SNMAD', 
                                               'B_MEDIAN', 'B_MIN', 'B_MAX',
                                               'SKY_AREA', 'NSRC', 'NFIELDS'])    

    bins_filename = '{}_optbins.fits'.format(nir_survey.lower())
    optbins_table.write(os.path.join(data_folder, bins_filename), 
                        format='fits', overwrite=True)
    
    return obsids_table


def xrays(obsids_table, min_fields, binid_start) :
    
    obsids_table.sort('EP_TEXP')
    nbins = int(len(obsids_table)/int(min_fields))
    stats = np.full((nbins, 8), np.nan)
    binid = np.full((len(obsids_table),), np.nan)
    
    j = 0
    for i in range(nbins) :
        if i == nbins - 1 :
            bin_table = obsids_table[j:]
            binid[j:] = binid_start + i
        else :
            bin_table = obsids_table[j:j+min_fields]            
            binid[j:j+min_fields] = binid_start + i

        avetexp = np.median(bin_table['EP_TEXP'])
        mintexp = np.min(bin_table['EP_TEXP'])
        maxtexp = np.max(bin_table['EP_TEXP'])
        area = np.sum(bin_table['SKY_AREA'])
        nsrc = np.sum(bin_table['NSRC_XMM'])
        stats[i,:] = [binid_start + i, avetexp, mintexp, maxtexp, 
                      nsrc, area, nsrc/area, len(bin_table)]

        j = j+min_fields
        
    stats_table = Table(stats, names=['BIN_ID', 'MEDIAN_TEXP', 'MIN_TEXP', 'MAX_TEXP',
                                      'NSRC_XMM', 'SKY_AREA', 'SKY_DENSITY_XMM', 'NFIELDS'])

    binid_col = Table.Column(binid, name='BIN_ID')
    obsids_table.add_column(binid_col)

    return obsids_table, stats_table

#    plt.loglog(obsids_table['EP_TEXP'], 
#                 obsids_table['NSRC_XMM']/obsids_table['SKY_AREA'],
#                 marker='.', ms=5, lw=0)
#    xerrmin = stats[:,0]-stats[:,1]
#    xerrmax = stats[:,2]-stats[:,0]
#    plt.errorbar(stats[:,0], stats[:,5], xerr=(xerrmin, xerrmax), 
#                 fmt='ro', capsize=5, zorder=1000)
#    plt.show()


def final(obsids_table, data_folder, nir_survey='2MASS') : 
    
    ### Get optical bins
    optbins = np.unique(obsids_table['OPTBIN'])
    
    obsids_table_bins = Table()
    stats = Table()

    ### Define Texp bins for each optical bin
    binid_first = 1
    for obin in tqdm(optbins, desc='Binning OBSIDs') :
        msk_bin = np.logical_and(obsids_table['OPTBIN'] == obin,
                                 ~obsids_table['SKY_OUTLIER'])
                                 
        bin_table = obsids_table[msk_bin]
        bin_table, bin_stats = xrays(bin_table, 7, binid_first)
        
        obsids_table_bins = vstack([obsids_table_bins, bin_table])
        stats = vstack([stats, bin_stats])
            
        binid_first += len(bin_stats)

    stats_filename = '{}_bins.fits'.format(nir_survey.lower())
    stats.write(os.path.join(data_folder, stats_filename), 
                format='fits', overwrite=True)
    
    msk_outliers = obsids_table['SKY_OUTLIER']
    outliers_table = obsids_table[msk_outliers]
    outliers_col = Table.Column(len(outliers_table)*[np.nan], name='BIN_ID')
    outliers_table.add_column(outliers_col)

    obsids_table_bins = vstack([obsids_table_bins, outliers_table])
        
    return obsids_table_bins
            

def makebins(obsids_table, data_folder, desctag, nir_survey='2MASS', 
             bincol='BIN_ID', errtype='circle') :
    
    bins = np.unique(obsids_table[bincol])
    bins = bins[~np.isnan(bins)]

    groups_folder = os.path.join(data_folder, 'groups')
    bins_folder = os.path.join(data_folder, 'bins')
    if nir_survey is not '2MASS' :
        bins_folder += '_{}'.format(nir_survey.lower())
        groups_folder += '_{}'.format(nir_survey.lower())

    if not os.path.exists(bins_folder) :
        os.makedirs(bins_folder)

    for binid in tqdm(bins, desc='Making {} bins'.format(desctag)) :

        bin_filename = os.path.join(
                        bins_folder,
                        'bin{}.fits'.format(str(int(binid)).zfill(3)))

        msk_bin = obsids_table[bincol] == binid
        bin_table = obsids_table[msk_bin]
        tables_array = [None]*len(bin_table)
        
        for i, obs in enumerate(bin_table) :
            group_file = os.path.join(groups_folder, 
                                      '{}.fits'.format(obs['OBS_ID']))            
            tables_array[i] = Table.read(group_file)
        
        bin_srcs = vstack(tables_array)
        bin_srcs.meta['AREA'] = np.sum(bin_table['SKY_AREA'])
        bin_srcs.meta['errortype'] = errtype
        
        bin_srcs.write(bin_filename, overwrite=True)

