#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jeb@geus.dk

obtain monthly coordinates from transmissions

input:
    aws data from Thredds server
    
output:
    daily aws albedo, lat, lon, and cloud index

"""
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from datetime import datetime
import sys
import simplekml
# import geopy.distance
from datetime import date
import calendar

# -------------------------------- set the working path automatically
if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/S3/SICE_ESSD/'
os.chdir(base_path)
sys.path.append(base_path)

# ----------------------------------------------------------
# ----------------------------------------------------------
# some main user-defined variables
plot_individual=0 ; site='QAS_Lv3' #'SWC_O'# 'QAS_U' 
do_plot=1 # set to 1 if user wants plots
plt_map=0 # if set to 1 draws a map to the right of the graphic
ly='x' # either 'x' or 'p', 'x' is for display to local plots window, 'p' writes a .png graphic
do_NRT=1 # do near realtime? 1 f or yes
do_ftp=0 # set to 1 if push values
plot_stars_on_extreme_values=1 # like it says ;-)
open_fig_testing=0
n_std=1.96 # 1.96 sigma corresponds to 95%ile
min_years_of_record_for_robust_stats=3
plt_last_val_text=1

# get today's date used in naming graphics and help with NRT
today = date.today()
today = pd.to_datetime(date.today())#-timedelta(days=1) 
versionx= today.strftime('%Y-%m-%d')
current_month=today.strftime('%m')
current_year=today.strftime('%Y')

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
# print(meta.columns)

names=meta.name

if plot_individual:
    names=names[names==site]


n_sites=len(names)

timeframe='day'
# timeframe='hour'
iyear=2017 ; fyear=2023
n_years=fyear-iyear+1


# names=['SDL']
# site='SDL'
# timeframe='hour'
for i,name in enumerate(names):
    if i>=0:
    # if i>=31:

        print()
        print(i,n_sites-i,name)#[i],meta.name[i],names[i][0:5])
        
        site=name

        # fn=aws_data_path+site+'/'+site+'_day.csv'
        # df=pd.read_csv(fn)        
        # print(site)
        url = "https://thredds.geus.dk/thredds/fileServer/aws_l3_station_csv/level_3/{}/{}_{}.csv".format(site,site,timeframe)
        df = pd.read_csv(url)
        v=~np.isnan(df['gps_lat'])
        # print(df['gps_lat'][v])
        # print(df['gps_lon'][v])
        # print(df['gps_alt'][v])
        # print(df)
        
        # print(df.columns)

        n=20
        lat=np.nanmean(df.gps_lat)#[-n:])
        lon=-abs(np.nanmean(df.gps_lon))#[-n:])
        elev=np.nanmean(df.gps_alt)
        # print(lat,lon,elev)
        
        # df.index = pd.to_datetime(df.time)
        # fig, ax = plt.subplots(figsize=(10,10))
        # plt.plot(df.batt_v)
        # plt.ylabel('VDC')
        # plt.title(name+' 2023 from Thredds')
        # plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )

        df["date"] = pd.to_datetime(df.time)

        df['year'] = pd.DatetimeIndex(df["date"]).year
        df['day'] = pd.DatetimeIndex(df["date"]).day
        df['hour'] = pd.DatetimeIndex(df["date"]).hour
        df['month'] = pd.DatetimeIndex(df["date"]).month
        df['doy'] = pd.DatetimeIndex(df["date"]).dayofyear
        df['doy_dec'] = df['doy']+(df['hour']-1)/23
        df.index = pd.to_datetime(df.time)
    
        # print(df.columns)
        # print(df)
        df=df[df.year>=2017]
        df=df[~np.isnan(df.albedo)]
        
        v=np.isnan(df.gps_lat)
        if sum(v)>0:
            print('missing lats',sum(v),meta.lat.values[i],meta.lon.values[i])
            df.gps_lat[v]=meta.lat.values[i]
            df.gps_lon[v]=meta.lon.values[i]

        
        # df=df[~np.isnan(df.gps_lat)]
        # df=df[~np.isnan(df.gps_lon)]
        df=df[df.month>=5]
        df=df[df.month<=9]
        
        # lat=df.gps_lat
        # lon=df.gps_lon
        # elev=df.gps_alt

        plt.close()
        fig, ax = plt.subplots(4,1,figsize=(8,12))

        cc=0
        ax[cc].plot(df['date'],df['gps_lat'],'.',color='k')
        ax[0].set_title(name+' latitude')
        ax[cc].set_xticklabels([])

        cc+=1
        ax[cc].plot(df['date'],df['gps_lon'],'.',color='k')
        ax[cc].set_title(name+' longitude')
        ax[cc].set_xticklabels([])

        cc+=1
        ax[cc].plot(df['date'],df['gps_alt'],'.',color='k')
        ax[cc].set_title(name+' elevation')
        plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
        ax[cc].set_xticklabels([])

        cc+=1
        ax[cc].plot(df['date'],df['cc'],'.',color='k')
        ax[cc].set_title(name+' cloud')
        plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
        
        figpath='./Figs/positions/'
        os.system('mkdir -p '+figpath)
        
        ly='p'
        
        if ly == 'p':
            plt.savefig(f'{figpath}{name}_lat_lon_elev.png', dpi=150, bbox_inches="tight", pad_inches=0.1)
            
        if site=='SDL':
            x=df['date'][df['gps_lat']>69]
            # print('last date with CP1 latitude',x[-1])
            inv=df['gps_lat']>69
            df['gps_lat'][x]=np.nan
            df['gps_lon'][x]=np.nan
            df['gps_alt'][x]=np.nan

        if site=='CP1':
            x=df['gps_alt']<1944
            df['gps_lat'][x]=np.nan
            df['gps_lon'][x]=np.nan
            df['gps_alt'][x]=np.nan

        df['alb_'+site]=df.albedo
        df['lat_'+site]=df.gps_lat
        df['lon_'+site]=df.gps_lon
        df['cloud_'+site]=df.cc
        df.to_csv('./data/AWS/'+name+'.csv',columns=['alb_'+site,'lat_'+site, 'lon_'+site,'cloud_'+site])


#%% not needed after all since QAS_L had all dates
# from datetime import date, timedelta

# def datesx(date0,date1):
#     # difference between current and previous date
#     delta = timedelta(days=1)
#     # store the dates between two dates in a list
#     dates = []
#     while date0 <= date1:
#         # add current date to list by converting  it to iso format
#         dates.append(date0.isoformat())
#         # increment start date by timedelta
#         date0 += delta
#     # print('Dates between', date0, 'and', date1)
#         # print(dates)
#     return dates

# years=np.arange(2017,2023+1).astype(str)
# # years=np.arange(2017,2017+1).astype(str)

# datesxx=[]

# for year in years:
#     print(year)
#     dates=datesx(date(int(year), 5, 1),date(int(year), 9, 30))
#     # print(dates)
#     for datey in dates:
#         datesxx.append(datey)

# datesxx=np.array(datesxx)
# print(len(datesxx))
#%%
# from glob import glob
# files=glob('./data/AWS/*')


df=pd.read_csv('./data/AWS/QAS_L.csv')
df = df.rename({'stid': 'name'}, axis=1)

# for i,file in enumerate(files):
for i,name in enumerate(names):
    print(i,name)
    df2=pd.read_csv('./data/AWS/'+name+'.csv')
    df=df.merge(df2,on='time', how='left', indicator=False)

#%%
df = df.rename({'alb_QAS_L_x': 'alb_QAS_L'}, axis=1)
df = df.rename({'lat_QAS_L_x': 'lat_QAS_L'}, axis=1)
df = df.rename({'lon_QAS_L_x': 'lon_QAS_L'}, axis=1)
df = df.rename({'cloud_QAS_L_x': 'cloud_QAS_L'}, axis=1)

df.to_csv('./data/AWS/all.csv')
