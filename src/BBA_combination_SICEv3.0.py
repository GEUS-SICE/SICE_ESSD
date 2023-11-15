#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:24:33 2023

@author: jason
"""


import xarray as xr
from pyproj import CRS,Transformer
import os
import numpy as np
from rasterio.transform import Affine
import rasterio as rio
import rasterio
from pathlib import Path
from datetime import date, timedelta

if os.getlogin() == 'jason':
    base_path='/Users/jason/Dropbox/S3/SICE_ESSD/'
    output_base_path = '/Users/jason/0_dat/S3/opendap/'

os.chdir(base_path)

version_index=1
resolution=['1000','500']

def read_S3(fn):
    test_file = Path(fn)
    # print(fn)
    r = np.zeros((5424, 2959)) * np.nan
    if test_file.is_file():
        print("reading " + fn)
        rx = rasterio.open(fn)
        r = rx.read(1)
        r[r > 1] = np.nan
    else:
        print("no file")
    return r


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
    # print('Dates between', date0, 'and', date1)
    # print(dates)
    return dates


def BBA_combination(write_out,
                    fn_band_1,fn_band_2,fn_band_3,fn_band_4,
                    fn_albedo_bb_planar_sw,
                    ofile):
    
    BBA_combo=np.zeros((5424, 3007))*np.nan

    test_file = Path(fn_band_1)
    if test_file.is_file():
        band_1x = rasterio.open(fn_band_1)
        profile=band_1x.profile
        band_1=band_1x.read(1)
    
        band_2x = rasterio.open(fn_band_2)
        # profile=band_2x.profile
        band_2=band_2x.read(1)

        band_3x = rasterio.open(fn_band_3)
        # profile=band_3x.profile
        band_3=band_3x.read(1)

        band_4x = rasterio.open(fn_band_4)
        # profile=band_4x.profile
        band_4=band_4x.read(1)

        albedo_bb_planar_swx = rasterio.open(fn_albedo_bb_planar_sw)
        # profile=albedo_bb_planar_swx.profile
        albedo_bb_planar_sw=albedo_bb_planar_swx.read(1)
    
        thresh_bare_ice=0.565

        a=1.003 ; b=0.058
        
        temp=a*((band_1+band_2+band_3+band_4)/4)+b

        v=temp<=thresh_bare_ice
        BBA_combo[v]=temp[v]
        
        v=temp>thresh_bare_ice
        BBA_combo[v]=albedo_bb_planar_sw[v]
        
        inv=BBA_combo>0.95
        BBA_combo[inv]=np.nan
        
        inv=BBA_combo<0.058
        BBA_combo[inv]=np.nan
        
        if write_out:
            with rasterio.Env():
                with rasterio.open(ofile, 'w', **profile) as dst:
                    dst.write(BBA_combo, 1)
    return None



years=np.arange(2017,2023+1).astype(str)
years=np.arange(2017,2019+1).astype(str)
years=np.arange(2019,2024).astype(str)
# years=np.arange(2017,2018).astype(str)
# years=np.arange(2018,2019).astype(str)
# years=np.arange(2020,2021).astype(str)
# years=np.arange(2021,2022).astype(str)
# years=np.arange(2022,2023).astype(str)
years=np.arange(2023,2024).astype(str)
 
for year in years:

    # dates=datesx(date(int(year), , 1),date(int(year), 8, 31))
    # dates=datesx(date(int(year), 7, 22),date(int(year), 7, 22))
    # dates=datesx(date(int(year), 8, 2),date(int(year), 8, 2))
    dates=datesx(date(int(year), 5, 1),date(int(year), 9, 30))
    dates=datesx(date(int(year), 9, 25),date(int(year), 9, 30))
    
    
    for datex in dates:
        print('BBA combo',datex)
        yearx=datex.split('-')[0]
        BBA_combination(1,
                        f'/Users/jason/0_dat/S3/opendap/Greenland_500m/{yearx}/{datex}_r_TOA_01.tif',
                        f'/Users/jason/0_dat/S3/opendap/Greenland_500m/{yearx}/{datex}_r_TOA_06.tif',
                        f'/Users/jason/0_dat/S3/opendap/Greenland_500m/{yearx}/{datex}_r_TOA_17.tif',
                        f'/Users/jason/0_dat/S3/opendap/Greenland_500m/{yearx}/{datex}_r_TOA_21.tif',
                        f'/Users/jason/0_dat/S3/opendap/Greenland_500m/{yearx}/{datex}_albedo_bb_planar_sw.tif',
                        f'/Users/jason/0_dat/S3/opendap/Greenland_500m/{yearx}/{datex}_BBA_combination.tif'
                        )