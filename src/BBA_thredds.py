# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:28:13 2024

@author: rabni
"""
import argparse
from pyproj import Transformer
import sys 
import logging
import time
import datetime
import glob
import shutil
import os
import requests 
import numpy as np
from scipy.ndimage import gaussian_filter
import json
import requests 
import pandas as pd
from rasterio.transform import Affine
import netCDF4 as nc 
import rasterio as rio
from pyproj import CRS as CRSproj
import warnings
import xarray as xr
from bs4 import BeautifulSoup
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
src_path = os.getcwd()
base_path = os.path.abspath('..')
crs_polar = CRSproj.from_string("+init=EPSG:3413")

def exporttiff(x,y,z,crs,path):
    
    "Input: xgrid,ygrid, data paramater, the data projection, export path, name of tif file"
    
    resx = (x[0,1] - x[0,0])
    resy = (y[1,0] - y[0,0])
    transform = Affine.translation((x.ravel()[0]),(y.ravel()[0])) * Affine.scale(resx, resy)
    
    if resx == 0:
        resx = (x[0,0] - x[1,0])
        resy = (y[0,0] - y[0,1])
        transform = Affine.translation((y.ravel()[0]),(x.ravel()[0])) * Affine.scale(resx, resy)
    
    with rio.open(
    path,
    'w',
    driver='GTiff',
    height=z.shape[0],
    width=z.shape[1],
    count=1,
    dtype=z.dtype,
    crs=crs,
    transform=transform,
    ) as dst:
        dst.write(z, 1)
    
    dst.close()

def files_http(url, ext='nc'):
    r = requests.Session().get(url).text
    soup = BeautifulSoup(r, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


def BBA_combination(ds_id):
    
    thresh_bare_ice = 0.565
    a = 1.003
    b = 0.058
    
    ds = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICE_500m/Greenland/{ds_id}')
    alb = np.array(ds['albedo_bb_planar_sw'])
    BBA_combo = np.ones_like(alb) * np.nan
 
    temp = np.array(a * ((ds['r_TOA_01'] + ds['r_TOA_06'] + ds['r_TOA_17'] + ds['r_TOA_21']) / 4) + b)
    
    v = temp <= thresh_bare_ice
    BBA_combo[v] = temp[v]
    
    v = temp > thresh_bare_ice
    BBA_combo[v] = alb[v]
    
    inv = BBA_combo > 0.95
    BBA_combo[inv] = np.nan

    inv = BBA_combo < 0.058
    BBA_combo[inv] = np.nan
    
    return BBA_combo


if __name__ == "__main__":
   
    region = 'Alaska_Yukon'
    base_url = 'https://trhedds.geus.dk'
    
    link = base_url + os.sep + region
    
    get_files_url = files_http(link)
    
    ds_start = ds = xr.open_dataset(get_files_url[0])
    x_grid,y_grid = np.meshgrid(ds_start.x,ds_start.y)
    
    for ds in get_files_url:
        bba = BBA_combination(ds)
        f_name = ds.split(os.sep)[-1].replace('nc','tif')
        exporttiff(x_grid, y_grid, bba, crs_polar, f_name)