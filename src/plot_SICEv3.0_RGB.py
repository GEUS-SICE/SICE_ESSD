#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 06:23:44 2023

@author: jason

outputs npy arrays and figures of SICE3 data

['ANG', 'AOD_550', 'O3_SICE', 'al', 'albedo_bb_planar_sw', 'albedo_bb_spherical_sw', 'albedo_spectral_planar_01',
 'albedo_spectral_planar_02', 'albedo_spectral_planar_03', 'albedo_spectral_planar_04', 'albedo_spectral_planar_05',
 'albedo_spectral_planar_06', 'albedo_spectral_planar_07', 'albedo_spectral_planar_08', 'albedo_spectral_planar_09',
 'albedo_spectral_planar_10', 'albedo_spectral_planar_11', 'albedo_spectral_planar_16', 'albedo_spectral_planar_17',
 'albedo_spectral_planar_18', 'albedo_spectral_planar_19', 'albedo_spectral_planar_20', 'albedo_spectral_planar_21',
 'cloud_mask', 'crs', 'cv1', 'cv2', 'factor', 'grain_diameter', 'isnow', 'lat', 'lon', 'r0', 'rBRR_01', 'rBRR_02',
 'rBRR_03', 'rBRR_04', 'rBRR_05', 'rBRR_06', 'rBRR_07', 'rBRR_08', 'rBRR_09', 'rBRR_10', 'rBRR_11', 'rBRR_16', 
 'rBRR_17', 'rBRR_18', 'rBRR_19', 'rBRR_20', 'rBRR_21', 'r_TOA_01', 'r_TOA_02', 'r_TOA_03', 'r_TOA_04', 'r_TOA_05',
 'r_TOA_06', 'r_TOA_07', 'r_TOA_08', 'r_TOA_09', 'r_TOA_10', 'r_TOA_11', 'r_TOA_12', 'r_TOA_13', 'r_TOA_14', 'r_TOA_15',
 'r_TOA_16', 'r_TOA_17', 'r_TOA_18', 'r_TOA_19', 'r_TOA_20', 'r_TOA_21', 'saa', 'snow_specific_surface_area', 'sza',
 'threshold', 'vaa', 'vza']
"""

import os
import pandas as pd
# from datetime import datetime 
import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.basemap import Basemap
import numpy as np
# import xarray as xr
import os.path
from glob import glob
import rasterio
from datetime import date, timedelta
from pathlib import Path

fs=24 ; th=1
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "grey"
plt.rcParams["font.size"] = fs
#params = {'legend.fontsize': 20,
#          'legend.handlelength': 2}
plt.rcParams['legend.fontsize'] = fs*0.8

def datesx(date0,date1):
    # difference between current and previous date
    delta = timedelta(days=1)
    # store the dates between two dates in a list
    dates = []
    while date0 <= date1:
        # add current date to list by converting  it to iso format
        dates.append(date0.isoformat())
        # increment start date by timedelta
        date0 += delta
    print('Dates between', date0, 'and', date1)
    print(dates)
    return dates

def read_S3(fn):
    test_file = Path(fn)
    # print(fn)
    r = np.zeros((5424, 3007)) * np.nan
    if test_file.is_file():
        print("reading " + fn)
        rx = rasterio.open(fn)
        r = rx.read(1)
        r[r > 1] = np.nan
    else:
        print("no file")
    return r

from skimage import exposure # maybe add this to the import packages block at the top?

def RGBx(f_Red,f_Green,f_Blue, out_file):
    red=read_S3(f_Red)
    gre=read_S3(f_Green)
    blu=read_S3(f_Blue)
    
    vred=red<0
    vgre=gre<0
    vblu=blu<0
    red[vred]=np.nan
    red[vgre]=np.nan
    red[vblu]=np.nan
    gre[vred]=np.nan
    gre[vgre]=np.nan
    gre[vblu]=np.nan
    blu[vred]=np.nan
    blu[vgre]=np.nan
    blu[vblu]=np.nan
    
    vred=red>1
    vgre=gre>1
    vblu=blu>1
    red[vred]=np.nan
    red[vgre]=np.nan
    red[vblu]=np.nan
    gre[vred]=np.nan
    gre[vgre]=np.nan
    gre[vblu]=np.nan
    blu[vred]=np.nan
    blu[vgre]=np.nan
    blu[vblu]=np.nan

    # v=((red<0)or(gre<0)or(blu<0))
    # red[v]=np.nan
    # gre[gre<0]=np.nan
    # blu[blu<0]=np.nan
#                v=np.where(red >=0 and red <=1)
    img = np.dstack((red,gre,blu))  # stacks 3 h x w arraLABELS -> h x w x 3
    # img[land] = exposure.adjust_log(img[land], 1.)
#                    # Gamma
    img = exposure.adjust_gamma(img, 2)
    
    # export as geoTIFF:
    bands = [red, gre, blu]
    
    red_o = rasterio.open(f_Red)
    meta = red_o.meta.copy()
    meta.update({"count": 3,
                 "nodata": -9999,
                 "compress": "lzw"})
    
    with rasterio.open(out_file, "w", **meta) as dest:
        for band, src in enumerate(bands, start=1):
            dest.write(src, band)
    
    return img    

raw_data_path='/Users/jason/0_dat/S3/opendap/Greenland_500m/'

doys=pd.to_datetime(np.arange(1,366), format='%j')
months=doys.strftime('%b')
months_int=doys.strftime('%m').astype(int)


years=np.arange(2017,2018).astype(str)
# years=np.arange(2022,2023).astype(str)
years=np.arange(2018,2019).astype(str)
years=np.arange(2023,2024).astype(str)
# years=np.arange(2021,2022).astype(str)
# years=np.arange(2017,2024).astype(str)

do_cum=0
cum_name='' ; cum_name2='cumulative'

ni=5424 ; nj=3007

mask_file="/Users/jason/Dropbox/S3/masks/Greenland_500m.tiff"
mask = rasterio.open(mask_file).read(1)
ni = mask.shape[0] ; nj = mask.shape[-1]
# v = np.where(mask == 1)
v = np.where(mask > 0)
land=np.zeros((ni,nj))*np.nan
land[v]=1
print(ni,nj)


years=['2023']
region_name='Greenland'


for year in years:

    dates=datesx(date(int(year), 7, 24),date(int(year), 9, 30))
    dates=datesx(date(int(year), 8, 19),date(int(year), 9, 30))
    dates=datesx(date(int(year), 9, 20),date(int(year), 9, 25))
    dates=datesx(date(int(year), 7, 4),date(int(year), 7, 4))
    dates=datesx(date(int(year), 8, 27),date(int(year), 8, 27))
    # dates=datesx(date(int(year), 8, 15),date(int(year), 8, 30))
    # dates=datesx(date(int(year), 5, 25),date(int(year), 9, 25))
    # dates=datesx(date(int(year), 7, 1),date(int(year), 9, 25))
    # dates=datesx(date(int(year), 7, 15),date(int(year), 8, 30))
    # dates=datesx(date(int(year), 8, 23),date(int(year), 8, 23))
    
                    
    # varnam='r_TOA_21' ; lo=0 ; hi=1 ; extend='both' ; units='unitless'
    # varnam='albedo_bb_planar_sw' ; lo=0.3 ; hi=0.85 ; extend='both' ; units='unitless'
    # varnam='AOD_550' ; lo=0. ; hi=0.25 ; extend='both' ; units='unitless'
    for i,datex in enumerate(dates):
        # if i==0:
        if i>=0:
            path_raw=f'/Users/jason/0_dat/S3/opendap/{region_name}_500m/'

            temp=RGBx(f"{path_raw}/{year}/{datex}_r_TOA_08.tif",
          f"{path_raw}{year}/{datex}_r_TOA_06.tif",
          f"{path_raw}/{year}/{datex}_r_TOA_02.tif",
          f"{path_raw}/{year}/{datex}_r_TOA_RGB.tif")
        # else:
        #     print('missing',datex)
                
