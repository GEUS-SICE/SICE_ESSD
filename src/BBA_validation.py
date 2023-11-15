# -*- coding: utf-8 -*-
"""

daily AWS albedo versus various SICE retrievals

"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import polyfit
from scipy import stats
from scipy.stats import gaussian_kde


var='albedo_bb_planar_sw'
# var='BBA_combination'
res='Greenland_500m' ; ver='3.0'

# var='BBA_combination'
# res='Greenland_1000m' ; ver='2.3.2'

cld_thresh=0.2
# cld_thresh=0.4
cld_thresh=0.5
# cld_thresh=0.8
thresh_bare_ice=0.565


# -------------------------------- set the working path automatically
if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/S3/SICE_ESSD/'
os.chdir(base_path)


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


fn=f'./data/colocated_AWS_SICE/all_may-sept_2017-2023_{res}_{var}.csv'
os.system('ls -lF '+fn)
df=pd.read_csv(fn)
print(df.columns)



# xx=np.array([thresh_bare_ice-osy,thresh_bare_ice+osy])
# yy=-xx+BIT*2


# compute RMSD for snow
thresh_bare_ice=0.565
v=( (np.isfinite(df.alb_s3)) & (np.isfinite(df.alb_AWS)) & (df.alb_s3<1) & (df.alb_AWS<0.95) & (df.cloud<cld_thresh) & (df.alb_s3>thresh_bare_ice)& (df.alb_AWS>thresh_bare_ice) )
x=df.alb_s3[v] ; y=df.alb_AWS[v]

RMSD_snow=np.sqrt(np.mean((y-x)**2))
bias_snow=np.mean(y-x)

N_snow=len(y)
print('N %.0f'%N_snow)
print('snow bias %.3f'%bias_snow)
print('snow RMSD %.3f'%RMSD_snow)

# compute RMSD for bare ice
v=( (np.isfinite(df.alb_s3)) & (np.isfinite(df.alb_AWS)) & (df.alb_s3<1) & (df.alb_AWS<0.95) & (df.cloud<cld_thresh) & (df.alb_s3<=thresh_bare_ice)& (df.alb_AWS<=thresh_bare_ice) )
x=df.alb_s3[v] ; y=df.alb_AWS[v]

RMSD_ice=np.sqrt(np.mean((y-x)**2))
bias_ice=np.mean(y-x)
b_ice, m_ice = polyfit(x,y, 1)
coefs_ice=stats.pearsonr(x,y)
xx_ice=[np.min(x),np.max(x)] ; xx_ice=np.array(xx_ice)

N_ice=len(y)
print('N %.0f'%N_ice)
print('ice bias %.3f'%bias_ice)
print('ice RMSD %.3f'%RMSD_ice)

# all cases
v=( (np.isfinite(df.alb_s3)) & (np.isfinite(df.alb_AWS)) & (df.alb_s3<1) & (df.alb_AWS<0.95) & (df.cloud<cld_thresh) )
x=df.alb_s3[v]
y=df.alb_AWS[v]
names_z=df.site[v]

# y=y[x<0.95]
# x=x[x<0.95]

# plt.plot(x)

# plt.figure(figsize=(15,15))
# plt.scatter(x, y)#,color='gray', alpha=0.8, s=20, facecolors='none', edgecolors='gray', linewidths=1.8)



# plt.show()

# v=np.where(((np.isfinite(y))&(np.isfinite(x))))
# # plt.plot(x,y,'.',c=(0.8, 0., 0.),label=lab) 
b, m = polyfit(x,y, 1)
coefs=stats.pearsonr(x,y)

#  plot 

# graphics definitions
th=2 # line thickness
formatx='{x:,.3f}' ; fs=18
plt.rcParams["font.size"] = fs
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = "#C6C6C6"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th/2
plt.rcParams['axes.linewidth'] = 1

ly='x'

plt.close()
fig, ax = plt.subplots(figsize=(10, 10))

# plt.scatter(x,y,marker='.')
# Calculate the point density

# v=np.where(((np.isfinite(x))& (np.isfinite(y))))
# x=x[v] ; y=y[v]
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
ax.scatter(x, y, c=z, s=20)

xx0=0.1 ; xx1=0.98
ax.set_xlim(xx0,xx1)
ax.set_ylim(xx0,xx1)

ax.plot([xx0,xx1],[xx0,xx1],c='grey',linestyle='--',linewidth=th*1.5)

osy=0.2

# xx=np.array([thresh_bare_ice-osy,thresh_bare_ice+osy])
# yy=-xx+thresh_bare_ice*2
# -0.7+thresh_bare_ice*2
# -0.5+thresh_bare_ice*2

# ax.plot(xx,yy,c='k',linewidth=th)

do_fit=0

if do_fit:
    xx=[np.min(x),np.max(x)] ; xx=np.array(xx)
    plt.plot(xx, b + m * xx, '--',c='k',linewidth=2,label='fit')#', in-situ = %.3f'%m+' + %.3f'%b)
    # print((b + m * xx[0])-(b + m * xx[1]))
    plt.plot(xx_ice, b_ice + m_ice * xx_ice, '--',c='k',linewidth=2,label='fit')#', in-situ = %.3f'%m+' + %.3f'%b)
    
plt.xlabel(f'SICE v{ver} {var}',fontsize=fs)
plt.ylabel('in-situ albedo',fontsize=fs)

countx=np.zeros(n_AWS)

for j,name in enumerate(names):
    countx[j]=sum(names_z==name)
 
s=np.argsort(countx)
s = s[::-1]

namesx=names.values[s]
countx=countx[s]

dy=0.025
# dy=0.029
xx0=1.01 ; yy0=0.985
cc=0
mult=0.7

for j in range(len(namesx)):
    # countx=sum(names_z==name)
    # print(j,countx[j])
    if countx[j]>0:
        ax.text(xx0,yy0-dy*cc,namesx[j]+': %0.f'%countx[j]+' days',color='k',rotation=0,
                fontsize=fs*mult,
                        transform=ax.transAxes,zorder=20,ha='left') # ,bbox=props
        cc+=1

props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')

# annotate upper left
xx0=0.025 ; yy0=0.96 ; dy=0.035
mult=1.
cc=0
ax.text(xx0,yy0-dy*cc,'snow',color='k',rotation=0,
        fontsize=fs*mult,transform=ax.transAxes,zorder=20,ha='left') ; cc+=1
ax.text(xx0,yy0-dy*cc,'N = %0.f'%N_snow,color='k',rotation=0,
        fontsize=fs*mult,transform=ax.transAxes,zorder=20,ha='left') ; cc+=1
ax.text(xx0,yy0-dy*cc,f'bias %.2f'%bias_snow,color='k',rotation=0,
        fontsize=fs*mult,transform=ax.transAxes,zorder=20,ha='left') ; cc+=1
ax.text(xx0,yy0-dy*cc,f'RMSD %.2f'%RMSD_snow,color='k',rotation=0,
        fontsize=fs*mult,transform=ax.transAxes,zorder=20,ha='left') ; cc+=1

cc+=0.5

ax.text(xx0,yy0-dy*cc,'bare ice',color='k',rotation=0,
        fontsize=fs*mult,transform=ax.transAxes,zorder=20,ha='left') ; cc+=1
ax.text(xx0,yy0-dy*cc,'N = %0.f'%N_ice,color='k',rotation=0,
        fontsize=fs*mult,transform=ax.transAxes,zorder=20,ha='left') ; cc+=1
ax.text(xx0,yy0-dy*cc,f'bias %.2f'%bias_ice,color='k',rotation=0,
        fontsize=fs*mult,transform=ax.transAxes,zorder=20,ha='left') ; cc+=1
ax.text(xx0,yy0-dy*cc,f'RMSD  %.2f'%RMSD_ice,color='k',rotation=0,
        fontsize=fs*mult,transform=ax.transAxes,zorder=20,ha='left') ; cc+=1

# annotate lower right
xx0=0.98 ; yy0=0.06 ; dy=0.04
mult=1.
cc=0
ax.text(xx0,yy0-dy*cc,'N = %0.f'%np.sum(countx)+'\n'+'cloud probability index < %.2f'%np.sum(cld_thresh),
        color='k',rotation=0,
        fontsize=fs*mult,
                transform=ax.transAxes,zorder=20,ha='right',bbox=props)
# cc+=1
# ax.text(xx0,yy0-dy*cc,'cloud probability index < %.2f'%np.sum(cld_thresh),color='k',rotation=0,
#         fontsize=fs*mult,
#                 transform=ax.transAxes,zorder=20,ha='right')

# ----------- annotation
xx0=-0.08 ; yy0=-0.11
mult=0.8
co=0.4
cwd=__file__  #cwd=os.getcwd()
plt.text(xx0, yy0, cwd.replace('/Users/jason/Dropbox/S3/',''),
        fontsize=fs*mult,color=[co,co,co],rotation=0,
        transform=ax.transAxes,zorder=20,ha='left') # 
# plt.legend()

ly='p'

if ly == 'x':plt.show()

if ly == 'p':
    figname=f'./Figs/validation_daily_scatterplots/{var}_{ver}_{cld_thresh}.png' 
    plt.savefig(figname, bbox_inches='tight', dpi=150)
    