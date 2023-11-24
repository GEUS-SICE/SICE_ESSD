# -*- coding: utf-8 -*-
"""
Creation started Mar 28 2023

search below for ##! where you may need to make adjustments

@author: Rasmus Nielsen (rabni@geus.dk) and Jason Box (jeb@geus.dk)

code is to read SICE data from the GEUS Thredds server, main point is to gather just what variables are needed instead of gathering the entire product

dependencies are hopefully satisfied using:
    pip install opendap-protocol
    conda update --all
and
    conda install -c conda-forge netcdf4

Depending on the SICE version, a list of available variables is given here.

v2.3.2 data:
    ['lat',
     'lon',
     'crs',
     'albedo_bb_planar_sw',
     'BBA_combination',
     'diagnostic_retrieval',
     'num_scenes',
     'r_TOA_01',
     'r_TOA_06',
     'r_TOA_17',
     'r_TOA_21',
     'SCDA_final',
     'snow_specific_surface_area']

v3.0 data:
    ['ANG', 'AOD_550', 'O3_SICE', 'al', 'albedo_bb_planar_sw',
    'albedo_bb_spherical_sw', 'albedo_spectral_planar_01',
    'albedo_spectral_planar_02', 'albedo_spectral_planar_03',
    'albedo_spectral_planar_04', 'albedo_spectral_planar_05',
    'albedo_spectral_planar_06', 'albedo_spectral_planar_07',
    'albedo_spectral_planar_08', 'albedo_spectral_planar_09',
    'albedo_spectral_planar_10', 'albedo_spectral_planar_11',
    'albedo_spectral_planar_16', 'albedo_spectral_planar_17',
    'albedo_spectral_planar_18', 'albedo_spectral_planar_19',
    'albedo_spectral_planar_20', 'albedo_spectral_planar_21',
    'cloud_mask', 'crs', 'cv1', 'cv2', 'factor', 'grain_diameter',
    'isnow', 'lat', 'lon', 'r0', 'rBRR_01', 'rBRR_02', 'rBRR_03',
    'rBRR_04', 'rBRR_05', 'rBRR_06', 'rBRR_07', 'rBRR_08', 'rBRR_09',
    'rBRR_10', 'rBRR_11', 'rBRR_16', 'rBRR_17', 'rBRR_18', 'rBRR_19',
    'rBRR_20', 'rBRR_21', 'r_TOA_01', 'r_TOA_02', 'r_TOA_03',
    'r_TOA_04', 'r_TOA_05', 'r_TOA_06', 'r_TOA_07', 'r_TOA_08',
    'r_TOA_09', 'r_TOA_10', 'r_TOA_11', 'r_TOA_12', 'r_TOA_13',
    'r_TOA_14', 'r_TOA_15', 'r_TOA_16', 'r_TOA_17', 'r_TOA_18',
    'r_TOA_19', 'r_TOA_20', 'r_TOA_21', 'saa',
    'snow_specific_surface_area', 'sza', 'threshold', 'vaa', 'vza']

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

##! choose a data version
version_index=0
resolution=['1000','500']
version_number=['2.3.2','3.']
##! choose a region
region='NovayaZemlya'

# projection info
WGSProj = CRS.from_string("+init=EPSG:4326") # source projection
PolarProj = CRS.from_string("+init=EPSG:3413") # example of an output projection
wgs_data = Transformer.from_proj(WGSProj, PolarProj) 

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

def ExportGeoTiff(x,y,z,crs,path):
    
    "Input: xgrid,ygrid, data paramater, the data projection, export path, name of tif file"
    
    resx = (x[1] - x[0])
    resy = (y[1] - y[0])
    transform = Affine.translation((x[0]),(y[0])) * Affine.scale(resx, resy)
    
    # z[z>1.1]=np.nan
    with rio.open(
        path,
        'w',
        driver='GTiff',
        height=z.shape[0],
        width=z.shape[1],
        count=1,
        compress='lzw',
        dtype=z.dtype,
        # dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        ) as dst:
            dst.write(z, 1)
    
    return None 

##! choose year range
years=np.arange(2017,2023+1).astype(str)
# years=np.arange(2019,2020).astype(str)
# years=np.arange(2017,2018).astype(str)
# years=np.arange(2018,2019).astype(str)
# years=np.arange(2020,2021).astype(str)
# years=np.arange(2021,2022).astype(str)
# years=np.arange(2022,2023).astype(str)
# years=np.arange(2023,2024).astype(str)
 
for year in years:

    ##! choose date range
    dates=datesx(date(int(year), 8, 2),date(int(year), 8, 2))
    dates=datesx(date(int(year), 5, 1),date(int(year), 9, 30))
    dates=datesx(date(int(year), 5, 1),date(int(year), 5, 2))

    ##! choose what bands you need
    bands=['albedo_bb_planar_sw','factor']
    bands=['albedo_bb_planar_sw','rBRR_17','rBRR_21']
    bands=['albedo_bb_planar_sw','rBRR_01','rBRR_17','rBRR_21']
    bands=['rBRR_21','rBRR_17','rBRR_01'] # NDSI
    bands=['rBRR_08','rBRR_06','rBRR_04'] # RGB
    bands=['rBRR_01','rBRR_21']
    bands=['r_TOA_21','rBRR_21']
    # bands=['rBRR_01','rBRR_21','rBRR_10','rBRR_11']
    # bands=['r_TOA_01','r_TOA_10','r_TOA_11','r_TOA_21']
    # bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_21"]
    # bands = ["r_TOA_02", "r_TOA_03", "r_TOA_04", "r_TOA_05", "r_TOA_06", "r_TOA_07", "r_TOA_08", "r_TOA_09", "r_TOA_10", "r_TOA_11", "r_TOA_12", "r_TOA_21"]
    bands = ["rBRR_02", "rBRR_03", "rBRR_04", "rBRR_05", "rBRR_06", "rBRR_07", "rBRR_08", "rBRR_09", "rBRR_10", "rBRR_11", "rBRR_12", "rBRR_21"]
    bands = ["r_TOA_01", "r_TOA_06", "r_TOA_17", "r_TOA_21",'albedo_bb_planar_sw']
    # bands = ["r_TOA_05"]
    # bands = ["r_TOA_02", "r_TOA_06", "r_TOA_08", "r_TOA_21"]
    # bands=['r_TOA_10','r_TOA_11']

    # bands=['r_TOA_02','r_TOA_10','r_TOA_11','r_TOA_21']
    # bands=['albedo_bb_planar_sw','snow_specific_surface_area']
    # bands=['sza']
    # bands=['r_TOA_21']
    # bands=['r_TOA_04'] # greenish
    # bands=['r_TOA_06','r_TOA_08','r_TOA_02'] 
    # bands=['albedo_bb_planar_sw','r_TOA_06','r_TOA_08']
    
    bands=['BBA_combination']
    
    # ############### main code
    
    # west_x,north_y = wgs_data.transform(lon_w, lat_n)
    # east_x,south_y = wgs_data.transform(lon_e, lat_s)
    
    # y_slice = slice(int(north_y),int(south_y))
    # x_slice = slice(int(west_x),int(east_x))
    
    output_path=f'/Users/jason/0_dat/S3/opendap/{region}_{resolution[version_index]}m/'
    if not(os.path.exists(output_path) and os.path.isdir(output_path)):
        os.makedirs(output_path)
    output_path=f'/Users/jason/0_dat/S3/opendap/{region}_{resolution[version_index]}m/'+ year + os.sep
    if not(os.path.exists(output_path) and os.path.isdir(output_path)):
        os.makedirs(output_path)
        
    # loop over dates
    for d in dates:
        if version_index==0:
            DATASET_ID = f'SICEv2.3.2_{region}_1000m_' + d.replace('-', '_') + '.nc'
        else:
            DATASET_ID = 'sice_500_' + d.replace('-', '_') + '.nc'

        # print(d)

        for var in bands:
            ofile=output_path + d + '_' + var + '.tif'
            test_file = Path(ofile)
            print(f'gathering {d} {var}')
            
            if not(test_file.is_file()):
                fn=f'https://thredds.geus.dk/thredds/dodsC/SICE_{resolution[version_index]}m/{region}/{DATASET_ID}'
                # print(fn)
                
                try:
                    ds = xr.open_dataset(fn)
                    # ds.variables
                    # list(ds.keys())
                    yshape,xshape = np.shape(ds[var])
                    
                    if version_index:
                        if yshape != 5424: 
                            ds = ds.rename({'x2':'xcoor'})
                            ds = ds.rename({'y2':'ycoor'})        
                        else:
                            ds = ds.rename({'x':'xcoor'})
                            ds = ds.rename({'y':'ycoor'})
                        
                        data = ds[var]#.sel(ycoor=y_slice,xcoor=x_slice)
                        x = ds['xcoor']#.sel(xcoor=x_slice)
                        y = ds['ycoor']#.sel(ycoor=y_slice)
                    else:
                        x = ds['x']#.sel(xcoor=x_slice)
                        y = ds['y']#.sel(ycoor=y_slice)
                                            
                    # print(np.shape(ds[var]))
                    # data = data.where(data <= plotting_dict[v]['maxval'])
                    # data = data.where(data >= plotting_dict[v]['minval'])
                    
                    # z = data.to_numpy()
                    # x = x.to_numpy()
                    # y = y.to_numpy()
    
                    z = ds[var].to_numpy()
                    x = x.to_numpy()
                    y = y.to_numpy()
                                # 
                    ExportGeoTiff(x, y, z, PolarProj, ofile)
                    
                    ds.close()
            
                except:
                    print('no such data to gather on this server')
            else:
                print('local file already exists')
