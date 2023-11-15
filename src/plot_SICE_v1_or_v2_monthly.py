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

month='06'

os.chdir('/Users/jason/Dropbox/S3/SICE_ESSD/')

varnam='albedo_bb_planar_sw'
varnam='snow_specific_surface_area' ; varnam2='SSA, $m^2 kg^{-1}$'


months=['06','07','08']
months2=['June','July','August']
abc=['a)','b)','c)']

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



stattype='median'
# stattype='mean'

vers=['SICEvPTEP1','SICEvPTEP2']
vers=['SICEvPTEP1']
# vers=['SICEvPTEP2']

for vv,ver in enumerate(vers):
    
    if ver=='SICEvPTEP1':
        mask_file="/Users/jason/Dropbox/1km_grid2/mask_1km_1487x2687.tif"
        mask_file="/Users/jason/Dropbox/S3/masks/Greenland_1km.tiff"
    if ver=='SICEvPTEP2':
        mask_file="/Users/jason/Dropbox/S3/masks/Greenland_500m.tiff"

    mask = rasterio.open(mask_file).read(1)
    ni = mask.shape[0] ; nj = mask.shape[-1]
    # v = np.where(mask == 1)
    v = np.where(mask > 0)
    land=np.zeros((ni,nj))*np.nan
    land[v]=1
    # plt.imshow(mask)
    # plt.colorbar()

    fig, axs = plt.subplots(1, 3, layout='constrained',figsize=(15,8))
    for year in years:
        for mm,month in enumerate(months):
    
            fn = f'/Users/jason/0_dat/S3/{ver}/monthly/{varnam}/{varnam}_{year}_{month}_{stattype}.tif'
            my_file = Path(fn)
            
            if my_file.is_file():
                var = rasterio.open(fn).read(1)

                region='Greenland'

                # var[land==1]=-2
                var[((land==1)&(~np.isfinite(var)))]=-2
                                
                if varnam=='snow_specific_surface_area':
                    lo=0 ; hi=50
                    my_cmap = plt.cm.get_cmap('magma')
                else:
                    lo=0.2 ; hi=0.95
                    my_cmap = plt.cm.get_cmap('viridis')
    
                mult=0.1
                # fig, ax = plt.subplots(figsize=(ni*mult,nj*mult))
                # plt.close()
                # plt.clf()
                ax = axs[mm]
                my_cmap.set_under("#AA7748")  #  brown, land
                pcm=ax.imshow(var,vmin=lo,vmax=hi,cmap=my_cmap)
                # pcm=ax.imshow(var,vmin=lo,vmax=hi,cmap='seismic')
                # ax.set_title(f"{year} {months2[mm]}")
                ax.axis("off")

                # ----------- annotation
                xx0=0.0 ; yy0=0.96
                mult=0.95 ; co=0.
                # props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
                ax.text(xx0, yy0, f"{abc[mm]} {year} {months2[mm]}",
                        fontsize=fs*mult,color=[co,co,co],rotation=0,
                        transform=ax.transAxes,zorder=20,ha='left') # ,bbox=props
                
                if mm == 2:
                    # plt.suptitle(f"{varnam}")
    
                    cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
                    cbar=fig.colorbar(pcm, ax=ax, cax=cax)
                    cbar.ax.set_title(f'{ver}\n{stattype}\n{varnam2}\n')
    
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
    
    ly='p'

    if ly == 'x':plt.show()
    
    figpath=f'./Figs/{region}'
    os.system('mkdir -p '+figpath)
    
    if ly == 'p':
        plt.savefig(f'{figpath}/{varnam}_{year}_JJA_{ver}_{stattype}.png', dpi=150, bbox_inches="tight", pad_inches=0.1)