#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 08:27:35 2023

@author: jason

The May through September colocation of satellite retrievals is made daily and the nearest pixel is chosen

Thus, when the site is lacking GPS data, the albedo values are ignored.

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from pyproj import Transformer
import rasterio

def reproject_points(x,y,
    in_projection: str = "4326",
    out_projection: str = "3413"):

    # reprojecting AWS lat lon to EPSG 3413
    inProj = f"epsg:{in_projection}"
    outProj = f"epsg:{out_projection}"

    trf = Transformer.from_crs(inProj, outProj, always_xy=True)
    x_coords, y_coords = trf.transform(x, y)

    return x_coords, y_coords

def getval(lon, lat): 
    # function to output raster value at lat lon
    idx = dat.index(lon, lat, precision=1E-6)
    return z[idx]

# -------------------------------- set the working path automatically
if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/S3/SICE_ESSD/'
os.chdir(base_path)

# ----------------------------------------------------------

meta = pd.read_csv('./ancil/AWS_latest_locations.csv')
meta = meta.rename({'stid': 'name'}, axis=1)
# meta.drop(meta[meta.name=='Roof_GEUS'].index, inplace=True) # drop original time column
# meta.drop(meta[meta.name=='Roof_PROMICE'].index, inplace=True) # drop original time column
# meta.drop(meta[meta.name=='UWN'].index, inplace=True) # drop original time column
# meta.drop(meta[meta.name=='NUK_U'].index, inplace=True) # drop original time column
# drop some sites from the list, sites not transmitting in the interest of time
# names=['HUM','DY2','CEN1','CEN2','CP1','NAE','NAU','NEM','NSE','SDM','SDL']
names=['KAN_B','SUM','QAS_Uv3','NUK_U','UWN','Roof_PROMICE','Roof_GEUS','LYN_T','LYN_L','KPC_Lv3','KPC_Uv3','THU_L2','WEG_B','MIT','ZAK_L','ZAK_U','ZAK_A'] #
for name in names:
    meta.drop(meta[meta.name==name].index, inplace=True) # drop original time column

meta=meta.sort_values(by='name', ascending=True)

names=meta.name

n_AWS=len(names)
#%%

sensor='OLCI'
inpath='/Users/jason/0_dat/S3/opendap/' ; outpath='colocated_AWS_SICE'
var='albedo_bb_planar_sw'
var='BBA_combination'
ver='Greenland_500m'
ver='Greenland_1000m'

sensor='MODIS'
inpath='/Users/jason/0_dat/' ; outpath='colocated_AWS_'+sensor
Path('./data/'+outpath).mkdir(parents=True, exist_ok=True)

# !! adjust for the others
var='BSA'
ver='MCD43'

#%%
df=pd.read_csv('./data/AWS/all.csv')
N_AWS_days=len(df)

refl=np.zeros((n_AWS,N_AWS_days))

date_z=[]
site_z=[]
alb_z=[]
alb_AWS_z=[]
cloud_AWS_z=[]

# MODIS data are integer, so divide them by 1000
divisor=1.
if sensor=='MODIS':
    divisor=1000.

for i in range(N_AWS_days):
# for i in range(47):
    # if i==46:
    if i>=0:
    # if i<=149:

        # print(ver,var,df.time[i],N_AWS_days-i)
        datex=df.time[i]
        if sensor=='OLCI':
            fn=f"{inpath}{ver}/{datex.split('-')[0]}/{datex}_{var}.tif"
        if sensor=='MODIS':
            datex=df.time[i].replace('-','_')
            fn=f'{inpath}/{ver}/Albedo_BSA_shortwave_Greenland_{datex}.tif'
            print(fn)
            
        my_file = Path(fn)
        if my_file.is_file():
            dat = rasterio.open(fn)
            z = dat.read()[0]
            
            for j,name in enumerate(names):
                if j>=0:
                # if name=='KAN_U':
                # if i>=31:
                    if ~np.isnan(df['lon_'+name][i]):
                        x,y= reproject_points(df['lon_'+name][i],df['lat_'+name][i])
                        refl[j,i]=getval(x, y)
                        date_z.append(datex)
                        site_z.append(name)
                        alb_z.append(refl[j,i]/divisor)
                        alb_AWS_z.append(df['alb_'+name][i])
                        cloud_AWS_z.append(df['cloud_'+name][i])
                        # print(datex,name,df['lon_'+name][i],df['lat_'+name][i],refl[j,i],df['alb_'+name][i])
        else:
            print('no file',datex)

#%% output results
    
out=pd.DataFrame({'date':np.array(date_z),
                  'site':np.array(site_z).astype(str),
                  'alb_sat':np.array(alb_z).astype(float),
                  'alb_AWS':np.array(alb_AWS_z).astype(float),
                  'cloud':np.array(cloud_AWS_z).astype(float),
                  # 'j':sentence_list[:,2].astype(int)
                  })

# adjust data precision
vals=['alb_sat']
for val in vals:
    out[val] = out[val].map(lambda x: '%.3f' % x)
    
out.to_csv(f'./data/{outpath}/all_may-sept_2017-2023_{ver}_{var}.csv',index=None)

