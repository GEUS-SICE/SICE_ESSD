# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:53:31 2023

@author: rabni
"""


import numpy as np
from osgeo import gdal, osr, gdalconst
import os
import glob
from pyhdf.SD import SD, SDC
import re
from multiprocessing import Pool
from utils import merge_tiffs
from pyproj import CRS
import xarray as xr
from rasterio.transform import Affine
import rasterio as rio
import time
import warnings
# base_path = os.path.abspath('..')
# src_path = os.getcwd()
# mask_path = base_path + os.sep + 'mask'


def opentiff(filename):

    "Input: Filename of GeoTIFF File "
    "Output: xgrid,ygrid, data paramater of Tiff, the data projection"
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    da = rio.open(filename)
    proj = CRS(da.crs)

    elevation = np.array(da.read(1),dtype=np.float32)
    nx,ny = da.width,da.height
    x,y = np.meshgrid(np.arange(nx,dtype=np.float32), np.arange(ny,dtype=np.float32)) * da.transform

    da.close()

    return x,y,elevation,proj



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
    
    return None 
def np_to_geotiff(
    data: np.ndarray,
    geotransform: tuple,
    gtiff_filename: str,
) -> None:
    """
    Converts a numpy array along with georeference data to a geotiff file.

    :param data: 2D array of the variable to store
    :param geotransform: tuple with format (xmin, xres, xrot, ymax, yres, rot)
                         with xrot and yrot set to 0 if North is up
    :param gtill_filename: full output file name
    """

    # create raster file with one band
    dst_ds = gdal.GetDriverByName("GTiff").Create(
        gtiff_filename, data.shape[1], data.shape[0], 1, gdal.GDT_Float64
    )

    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    # srs.ImportFromEPSG(epsg)  # set CRS
    srs.ImportFromProj4("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(data)  # write band to the raster
    dst_ds.FlushCache()  # write to disk
    dst_ds = None

    return None

def set_raster_nan(source: str):
    x,y,z,proj = opentiff(source)
    z[z==0] = -999
    exporttiff(x, y, z, proj, source)
    return None
    
def raster_match(source: str, target: str, output: str) -> None:
    """
    Matches target raster to source independently of projection and resolution

    :param source: full file name of source/file to match
    :param target: full file name of target/reference file
    :param output: fill file name of converted source

    """
    # source
    src_filename = source
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()

    # raster to match
    match_filename = target
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # output/destination
    dst_filename = output
    dst = gdal.GetDriverByName("Gtiff").Create(
        dst_filename, wide, high, 1, gdalconst.GDT_Float32
    )
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    # run
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)

    del dst  # Flush

    return None


# extract data and convert to geotiff
# create circular kernel
def createKernel(radius):
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    return kernel


def MODIS_hdf_to_geotiff(file_names: str) -> None:
    """
    Reads MODIS hdf data, extracts georeferencing data, converts to geotiff, and
    matches to SICE data.

    :param file_name: full MODIS hdf file name

    """
    folder = r'C:\\Users\\rabni\\OneDrive - GEUS\\MODIS\\output'
    #file_names = glob.glob(f"{folder}/*.hdf")

    # if len(sorted(glob.glob(f"{folder}/*mosaic*"))) > 0:
    #     print("already processed, move on")
    #     return None

    for file_name in file_names:
        # open hdf file using the Scientific Dataset (SD) python API

        try:
            hdf = SD(file_name, SDC.READ)
        except:
            print(file_name, "********")
            return None

        # select albedo
        albedo_obj = hdf.select("Snow_Albedo_Daily_Tile")

        # get albedo data (integers)
        albedo_int = albedo_obj.get()
        # convert integers to float
        albedo_float = albedo_int / 100

        # dilate clouds by a 10-pixel radius (10*500m)
        # clouds_grow = ndimage.morphology.binary_dilation(
        #     albedo_int == 150, structure=createKernel(10)
        # )

        # mask albedo data for clouds (and land, actually)
        # albedo_float[clouds_grow] = np.nan
        albedo_float[albedo_int == 150] = np.nan

        # mask albedo data for quality flags
        albedo_float[albedo_int > 100] = np.nan

        fattrs = hdf.attributes(full=1)
        ga = fattrs["StructMetadata.0"]
        gridmeta = ga[0]

        # Construct the grid
        ul_regex = re.compile(
            r"""UpperLeftPointMtrs=\(
                                  (?P<upper_left_x>[+-]?\d+\.\d+)
                                  ,
                                  (?P<upper_left_y>[+-]?\d+\.\d+)
                                  \)""",
            re.VERBOSE,
        )
        match = ul_regex.search(gridmeta)
        x0 = float(match.group("upper_left_x"))
        y0 = float(match.group("upper_left_y"))

        lr_regex = re.compile(
            r"""LowerRightMtrs=\(
                                  (?P<lower_right_x>[+-]?\d+\.\d+)
                                  ,
                                  (?P<lower_right_y>[+-]?\d+\.\d+)
                                  \)""",
            re.VERBOSE,
        )
        match = lr_regex.search(gridmeta)
        x1 = float(match.group("lower_right_x"))
        y1 = float(match.group("lower_right_y"))
        ny, nx = albedo_float.shape
        xinc = (x1 - x0) / nx
        yinc = (y1 - y0) / ny

        # # create geotransform
        geotransform = (x1 - xinc * nx, xinc, 0, y1 - yinc * ny, 0, yinc)

        # generate output file name
        gtiff_filename = re.sub(".hdf", ".tif", file_name)

        if not os.path.exists(gtiff_filename) or os.path.getsize(gtiff_filename) == 0:
            # convert numpy array and associated geodata to geotiff
            np_to_geotiff(albedo_float, geotransform, gtiff_filename=gtiff_filename)

            # match newly created MODIS geotiff to 500m Greenland mask
            target = (
                mask_path + os.sep + 'AlaskaYukon_500m.tif'
            )

            match_filename = re.sub(".tif", "_500m.tif", gtiff_filename)
            raster_match(gtiff_filename, target, match_filename)
            #set_raster_nan(match_filename)

        else:
            print(file_name + "  not processed")

   
    mosaic_name = f"{file_names[0].rsplit(os.sep, 1)[0]}/{file_names[0].split(os.sep)[-1][:10]}_500m_mosaic.tif"
    
    file_tif_names = [f.replace(".hdf", "_500m.tif") for f in file_names]
    
    
    merge_tiffs(file_tif_names, mosaic_name,overwrite=True)

    tiffile_names = glob.glob(f"{folder}/*.tif")
    tiffile_names = [f for f in tiffile_names if not "mosaic" in f]

    for ff in tiffile_names:
        os.remove(ff)

    print(f'{mosaic_name} is done!!')

    return None


if __name__ == "__main__":
    
    
    
    while True:
        
        src_path = os.getcwd() 
        base_path = os.path.abspath('..')
        data_f = base_path + os.sep + 'output'
        mask_path = base_path + os.sep + 'mask'
        tif_files = glob.glob(data_f + os.sep + '*_500m_mosaic.tif')
        #tif_files = [f for f in tif_files if "mosaic" in f]
        tif_files_d = [f.split(os.sep)[-1][:10] for f in tif_files]
        files = glob.glob(data_f + os.sep + '*.hdf')
        dates = list(set(f.split(os.sep)[-1][:10] for f in files))
        if not tif_files_d:
            dates_not_done = dates
        else:
            dates_not_done = [d for d in dates if d not in tif_files_d]
        
        for i,d in enumerate(dates_not_done): 
            
            f_to_do = [f for f in files if d in f]
           
            if len(f_to_do) < 6:
                print(f'some data is missing on the this date: {d}, skipping')
                if i+1 == len(dates):
                    print('Waiting for download')
                    time.sleep(60)
                    break
                else:
                    continue
            else:
                print(f'processing {d}')
                MODIS_hdf_to_geotiff(f_to_do)
                for f in f_to_do:
                    if os.path.isfile(f):
                        os.remove(f)     
        
        print('Waiting for download,again')
        time.sleep(60)