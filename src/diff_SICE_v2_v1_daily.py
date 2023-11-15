# -*- coding: utf-8 -*-
"""
@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)
"""

from osgeo import gdal, gdalconst
import rasterio
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import calendar

year='2022'
month='06'

os.chdir('/Users/jason/Dropbox/S3/SICE_ESSD/')

varnam='albedo_bb_planar_sw'


months=['06','07','08']
months=['07']
months2=['June','July','August']
months2=['July']
# months=['08']
# years=['2018']

iyear=2022 ; fyear=2022
years=np.arange(iyear,fyear+1).astype(str)

th=1 ; fs=16
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams["font.size"] = fs
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th
plt.rcParams['axes.linewidth'] = th #set the value globally
plt.rcParams['figure.figsize'] = 17, 10

region='Greenland'

mask_file='/Users/jason/Dropbox/S3/masks/Greenland_1km.tiff'
mask = rasterio.open(mask_file).read(1)
ni = mask.shape[0] ; nj = mask.shape[-1]
v = np.where(mask == 1)
# v = np.where(mask > 0)
land=np.zeros((ni,nj))*np.nan
land[v]=1

# plt.imshow(land)

v1_daily_rasters_path = "/Users/jason/0_dat/S3/SICE_adrien/"
v2_daily_rasters_path = "/Users/jason/0_dat/S3/opendap/"


fig, axs = plt.subplots(1, 3, layout='constrained',figsize=(15,8))
# plt.close()
# plt.clf()




for year in years:
    for mm,month in enumerate(months):

        n_days=calendar.monthrange(int(year),int(month))[1]
        days=np.arange(1,n_days+1).astype(str)
        days=['14','16','31']
        abc=['a)','b)','c)']

        for dd,dayx in enumerate(days):
            # if dayx=='3': # 3 ok
            # if int(dayx)>=0: # 3 ok
            day=dayx.zfill(2)
        
            fn2=f"{v2_daily_rasters_path}{region}/{year}/{year}-{month}-{day}_{varnam}.tif"
            print(fn2)
            my_file2 = Path(fn2)
        
            fn1=f"{v1_daily_rasters_path}{region}/{year}-{month}-{day}/{varnam}.tif"
            print(fn1)
            my_file1 = Path(fn1)

            if ( (my_file1.is_file()) & (my_file2.is_file()) ):
                target_crs_fn='/Users/jason/Dropbox/1km_grid2/mask_1km_1487x2687.tif'
                ofile = f'/Users/jason/0_dat/S3/daily_comp_v1_v2/{varnam}_{year}_{month}_{day}_v2_1km.tif'
                
                profile = rasterio.open("/Users/jason/Dropbox/1km_grid2/mask_1km_1487x2687.tif").profile
    
                #source
                src = gdal.Open(fn2, gdalconst.GA_ReadOnly)
                src_proj = src.GetProjection()
                src_geotrans = src.GetGeoTransform()
                
                #raster to match
                match_ds = gdal.Open(target_crs_fn, gdalconst.GA_ReadOnly)
                match_proj = match_ds.GetProjection()
                match_geotrans = match_ds.GetGeoTransform()
                wide = match_ds.RasterXSize
                high = match_ds.RasterYSize
                
                #output/destination
                dst = gdal.GetDriverByName('Gtiff').Create(ofile, wide, high, 1, gdalconst.GDT_Float32)
                dst.SetGeoTransform( match_geotrans )
                dst.SetProjection( match_proj)
                
                #run
                # gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour) #.GRA_Bilinear
                gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
                del dst # Flush
                
                # os.system('ls -lF '+ofile)
                
                PTEPv1 = rasterio.open(fn1).read(1)
                PTEPv2 = rasterio.open(ofile).read(1)
                var=PTEPv1-PTEPv2
                var[land==1]=-2
                # var[((land==1)&(~np.isfinite(var)))]=-20
                
                region='Greenland'
                
                ly='p'

                lo=-0.15 ; hi=-lo
                mult=0.1
                # fig, ax = plt.subplots(figsize=(ni*mult,nj*mult))
                # fig, ax = plt.subplots(figsize=(10,10))
                # plt.close()
                # plt.clf()
                # ax.imshow(var,vmin=lo,vmax=hi,cmap='seismic')
                ax = axs[dd]
                my_cmap = plt.cm.get_cmap('seismic')
                my_cmap.set_under("#AA7748")  #  brown, land
                pcm=ax.imshow(var,vmin=lo,vmax=hi,cmap=my_cmap)
                # ax.set_title(f"{abc[dd]} {year} {months2[mm]} {day}")
                ax.axis("off")
                
                
                # ----------- annotation
                xx0=0.0 ; yy0=0.96
                mult=0.95 ; co=0.
                # props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
                ax.text(xx0, yy0, f"{abc[dd]} {year} {month} {day}",
                        fontsize=fs*mult,color=[co,co,co],rotation=0,
                        transform=ax.transAxes,zorder=20,ha='left') # ,bbox=props
                    # plt.suptitle(f"{varnam}")
                if dd == 2:        
                    cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
                    cbar=fig.colorbar(pcm, ax=ax, cax=cax)
                    cbar.ax.set_title(f'{varnam}\nPTEPv1\nminus\nPTEPv2\n')

                    # ----------- annotation
                    xx0=-0.1 ; yy0=-0.02
                    mult=0.8
                    co=0.4
                    cwd=__file__  #cwd=os.getcwd()
                    # props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
                    plt.text(xx0, yy0, cwd,
                            fontsize=fs*mult,color=[co,co,co],rotation=0,
                            transform=ax.transAxes,zorder=20,ha='left') # ,bbox=props
            else:
                print(f'no file {varnam}_{year}_{month}')

        # plt.colorbar()ax.set_title(f"{year} {months2[mm]}")
if ly == 'x':plt.show()

figpath=f'./Figs/{region}'
figpath='/Users/jason/0_dat/S3/daily_comp_v1_v2/Figs'
os.system('mkdir -p '+figpath)

figpath='/Users/jason/Dropbox/S3/SICE_ESSD/Figs/Greenland/'

if ly == 'p':
    plt.savefig(f'{figpath}/{varnam}_{year}_{month}__select_PTEPv1-v2.png', dpi=150, bbox_inches="tight", pad_inches=0.1)

