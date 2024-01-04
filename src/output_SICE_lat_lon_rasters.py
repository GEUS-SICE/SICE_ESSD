#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:13:55 2023

@author: jason
"""


#%%
            fn=f'https://thredds.geus.dk/thredds/dodsC/SICE_{resolution[version_index]}m/{region}/{DATASET_ID}'
            # print(fn)
            ds = xr.open_dataset('https://thredds.geus.dk/thredds/dodsC/SICE_500m/Greenland/sice_500_2023_07_04.nc')
            # #%%
            # lon,lat = np.meshgrid(np.array(ds['lon']),np.array(ds['lat']))
            # np.shape(np.array(lat))

            #%%

        
            x,y = np.meshgrid(np.array(ds['x']),np.array(ds['y']))
            transformer = Transformer.from_proj(PolarProj,WGSProj)
            lon,lat = transformer.transform(x,y)
            
            #%%


            ofile='/Users/jason/Dropbox/S3/SICE_ESSD/ancil/SICE_0.5km_lon_3007x5424.npz'
            b = lon.astype(np.float16)
            np.savez_compressed(ofile,lon=b)
            
            ofile='/Users/jason/Dropbox/S3/SICE_ESSD/ancil/SICE_0.5km_lat_3007x5424.npz'
            b = lat.astype(np.float16)
            np.savez_compressed(ofile,lat=b)
#%%
# lat_SICE=np.load('/Users/jason/Dropbox/S3/SICE_ESSD/ancil/SICE_0.5km_lat_3007x5424.npz') ; lat_SICE=lat_SICE['lat']
# lon_SICE=np.load('/Users/jason/Dropbox/S3/SICE_ESSD/ancil/SICE_0.5km_lon_3007x5424.npz') ; lon_SICE=lon_SICE['lon']
# plt.imshow(lon_SICE)
# plt.colorbar()