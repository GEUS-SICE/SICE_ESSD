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

def make_plot(do_plot,varnam,ly,plotvar,lo,hi):
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
    
    if varnam=='grain_diameter':
        cm = plt.cm.magma                        
        cm.set_over('r')
    else:
        cm = plt.cm.viridis
        cm.set_under('purple')
        cm.set_over('orange')
        
    c=plotvar

    # if varnam!='age':
    pp=plt.imshow(c,cmap=cm,
              vmin=lo,vmax=hi)
    # else:
    #     pp=plt.imshow(c,cmap=cm)        
    
    plt.axis('Off')
    # plt.colorbar()
    
    # ax = plt.gca()     
    
    xx0=0.99
    yy0x=0.18
    dyx=0.4
    
    
    # ---------------------    
    # --------------------- colorbar location
    cbaxes = fig.add_axes([xx0-0.04, yy0x, 0.03, dyx]) 
    
    if varnam!='age':
        cbar = plt.colorbar(pp,orientation='vertical',cax=cbaxes,extend=extend)
    else:
        cbar = plt.colorbar(pp,orientation='vertical',cax=cbaxes)

    # # --------------------- colorbar location
    # xx0=0.6 ; yy0x=0.1 ; dxy=0.4
    # cbaxes = fig.add_axes([xx0-0.04, yy0x, 0.015, dxy]) 
    
    # cbar = plt.colorbar(ax,orientation='vertical',format="%d",cax=cbaxes)
    # # cbar = plt.colorbar(ax,orientation='vertical')

    # mult=1
    # yy0=yy0x+0.45 ; dy2=-0.03 ; cc=0
    
    cc=0
    
    if do_cum:
        units_title=varnam+'\n'+cum_name2+',\n'+units
        tit=datex+'\n'+varnam+',\n'+cum_name2
    else:
        units_title=units
        tit=datex+'\n'+varnam
    
    plt.text(1.0, 0.87,tit, fontsize=fs*1.2,
             transform=ax.transAxes, color='k') ; cc+=1. 
    
    plt.text(xx0+0.15, yy0x+dyx+0.04,units_title,ha='center', fontsize=fs,
             transform=ax.transAxes, color='k') ; cc+=1. 
    
    plt.text(1.0, 0.005,'Sentinel-3 SICEv3\nGEUS, ESA NoR', fontsize=fs,
             transform=ax.transAxes, color='b') ; cc+=1. 
    
    
    if ly == 'x':
         plt.show() 
    
    DPI=100
     
    if ly == 'p':
         figpath='/Users/jason/0_dat/S3/opendap/Figs/'+varnam+'/'
         os.system('mkdir -p '+figpath)
         if do_cum:
             figpath='/Users/jason/0_dat/S3/opendap/Figs/'+varnam+'/'+cum_name2+'/'
         os.system('mkdir -p '+figpath)
         figname=datex+cum_name
         plt.savefig(figpath+figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')

if do_cum:
    plotvar=np.zeros((ni,nj))*np.nan
    cum_name='_cumu'
    cum_name2='cumulative'

age=np.zeros((ni,nj))#*np.nan


for year in years:

    dates=datesx(date(int(year), 7, 24),date(int(year), 9, 30))
    dates=datesx(date(int(year), 8, 19),date(int(year), 9, 30))
    dates=datesx(date(int(year), 9, 20),date(int(year), 9, 25))
    dates=datesx(date(int(year), 7, 8),date(int(year), 7, 15))
    dates=datesx(date(int(year), 8, 15),date(int(year), 8, 30))
    # dates=datesx(date(int(year), 5, 25),date(int(year), 9, 25))
    # dates=datesx(date(int(year), 7, 1),date(int(year), 9, 25))
    # dates=datesx(date(int(year), 7, 15),date(int(year), 8, 30))
    # dates=datesx(date(int(year), 8, 23),date(int(year), 8, 23))
    
                    
    varnam='r_TOA_21' ; lo=0 ; hi=1 ; extend='both' ; units='unitless'
    # varnam='albedo_bb_planar_sw' ; lo=0.3 ; hi=0.85 ; extend='both' ; units='unitless'
    # varnam='AOD_550' ; lo=0. ; hi=0.25 ; extend='both' ; units='unitless'
    for i,datex in enumerate(dates):
        # if i==0:
        if i>=0:
            # datex=pd.to_datetime(file.split('/')[-1][0:10]).strftime('%Y-%m-%d')
        
            fn=f"{raw_data_path}{year}/{datex}_{varnam}.tif"
            my_file = Path(fn)
    
            if (my_file.is_file()):
                print(datex)
                file = rasterio.open(fn)
                # profile_S3=SWIR1x.profile
                r=file.read(1)
                
    
                # print(ni,nj)
                
                if do_cum:
                    v=np.where(~np.isnan(r))
                    plotvar[v]=r[v]
                    age[v]+=1

                    # age[((land==1)&(~np.isfinite(var)))]=-2
                else:
                    plotvar=r
                age[np.isnan(r)]=0
                # r=ds.variables['albedo_spectral_planar_20'].values
                # r=ds.variables['rBRR_21'].values
                
                # lat=ds.variables['lon'].values
                # lon=ds.variables['lat'].values
                
            
                
                
                # subset_it=1 # subset south Greenland Q transect
                
                # if subset_it:
                #     xc0=ni-1000 ; xc1=ni # wider and taller
                #     yc0=nj-1000 ; yc1=nj
                
                #     xc0=0 ; xc1=ni-1000 # wider and taller
                #     yc0=0 ; yc1=nj-1000
                
                #     xc0=0 ; xc1=nj-1
                #     yc0=2500 ; yc1=ni-1
                #     yc0=1500 ; yc1=ni-1 # taller
                    
                #     nix=xc1-xc0+1
                #     njx=yc1-yc0+1
                # wo=0
                # if wo:
                #     opath='/Users/jason/0_dat/S3/SICE3/'+var+'/'
                #     os.system('mkdir -p '+opath)
                #     np.save(opath+datex+cum_name+'.npy', plotvar[yc0:yc1,xc0:xc1])#, allow_pickle=True, fix_imports=True)
            
                do_plot=1
                ly='p'

                make_plot(do_plot,'r_TOA_21',ly,plotvar,lo,hi)
                # make_plot(do_plot,'age',ly,age,0,8)
            else:
                print('missing',datex)
                
