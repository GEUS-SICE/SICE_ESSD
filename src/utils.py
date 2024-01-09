# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:10:45 2022

@author: rabni
"""
import os
import logging
import subprocess
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from pyproj import CRS
from affine import Affine
import xarray as xr
gm = r"C:\Users\rabni\Anaconda3\envs\GreenShift\Lib\site-packages\GDAL-3.0.2-py3.9-win-amd64.egg-info\scripts\gdal_merge.py"
from multiprocessing import Pool
from itertools import repeat

logger = logging.getLogger(__name__)


def OpenRaster(filename):
    
   "Input: Filename of GeoTIFF File "
   "Output: xgrid,ygrid, data paramater of Tiff, the data projection"
   
   
   
   da = xr.open_rasterio(filename)
   proj = CRS.from_string(da.crs)

   transform = Affine(*da.transform)
   elevation = np.array(da.variable[0],dtype=np.float32)
   nx,ny = da.sizes['x'],da.sizes['y']
   x,y = np.meshgrid(np.arange(nx,dtype=np.float32), np.arange(ny,dtype=np.float32)) * transform
   
   
   
   return x,y,elevation,proj



def ExportGeoTiff(x,y,z,crs,path,filename):
    
    "Input: xgrid,ygrid, data paramater, the data projection, export path, name of tif file"
    
    resx = (x[0,1] - x[0,0])
    resy = (y[1,0] - y[0,0])
    transform = Affine.translation((x.ravel()[0]),(y.ravel()[0])) * Affine.scale(resx, resy)
    
    if resx == 0:
        resx = (x[0,0] - x[1,0])
        resy = (y[0,0] - y[0,1])
        transform = Affine.translation((y.ravel()[0]),(x.ravel()[0])) * Affine.scale(resx, resy)
    
    with rasterio.open(
    path + os.sep + filename,
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
    
    return None 
    

def TemporalMerge(bandout,folder):
    
    files = glob.glob(folder + os.sep + "*.tif")
    
    if int(bandout[-1:]) == 1:
        filtfiles = [x for x in files if (int(x[-6:-4]) < 16)]
    else: 
        filtfiles = [x for x in files if (int(x[-6:-4]) > 16)]
    
    bandno = bandout[:3]
    bandFiles = [x for x in filtfiles if ((x[-13:-10]) == bandno)]
    
    start = 1
    
    for i,f in enumerate(bandFiles): 
        
        x,y,z,crs = OpenRaster(f)
         
        if start == 1: 
            
            data = np.tile(z * np.nan, (len(bandFiles), 1, 1))
            start = 0
            
        data[i,:,:] = z
        
    
    m,n = np.shape(data[0,:,:])
    
    merge = np.array([[np.nanmean(data[:,i,j]) for j in range(n)] for i in range(m)])
    merge[merge == 0] = np.nan
    
    ExportGeoTiff(x,y,merge,crs,folder,bandout + ".tif")
    
    for f in bandFiles:
        os.remove(f)
        
    return x,y,merge,crs

def merge_tiffs(input_filename_list, merged_filename, *, overwrite=False, delete_input=False):
    """Performs gdal_merge on a set of given geotiff images

    :param input_filename_list: A list of input tiff image filenames
    :param merged_filename: Filename of merged tiff image
    :param overwrite: If True overwrite the output (merged) file if it exists
    :param delete_input: If True input images will be deleted at the end
    """
    if os.path.exists(merged_filename):
        if overwrite:
            os.remove(merged_filename)
        else:
            raise OSError(f"{merged_filename} exists!")

    #logger.info("merging %d tiffs to %s", len(input_filename_list), merged_filename)
    
    #merge_command = ["python", gm, "-co", "compress=LZW", "-n", "0.0", "-o", merged_filename,  *input_filename_list]
    merge_command = ["python", gm] + ["-o", merged_filename] + ["-n", "0.0"] + input_filename_list + ["-co", "COMPRESS=LZW"] 
    subprocess.call(merge_command,shell=True)
    
    #subprocess.check_call(
    #    [r"C:\Users\rabni\Desktop\Work\GreenShift\gdal_merge.py", "-co", "BIGTIFF=YES", "-co", "compress=LZW", "-o", merged_filename, *input_filename_list]
    #)
    #logger.info("merging done")
    
    if delete_input:
        #logger.info("deleting input files")
        for filename in input_filename_list:
            if os.path.isfile(filename):
                os.remove(filename)     
                
def BiMonthlyMaps(folder):
    
    prod = ["B02_1","B03_1","B04_1",\
            "B02_2","B03_2","B04_2"]
        
        
    folder = [folder,folder,folder,\
            folder,folder,folder]
        
    #print(zip(prod,repeat(folder)))
    
    
    with Pool(len(prod)) as p:
        #p.starmap(TemporalMerge,zip(prod,repeat(folder)))
        p.starmap(TemporalMerge,zip(prod,folder))
        
        
    
def RasterMerge(inputfiles,outputname):
    
    raster_to_mosiac = []
   
    for p in inputfiles:
        raster = rasterio.open(p)
        raster_to_mosiac.append(raster)
        
    mosaic, output = merge(raster_to_mosiac)
    
    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
        }
    )
        
    with rasterio.open(outputname, "w", **output_meta) as m:
        m.write(mosaic)