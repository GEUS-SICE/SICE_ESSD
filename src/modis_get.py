 
import requests 
import pandas as pd
import os
import time
import glob
from bs4 import BeautifulSoup
    # url = "https://e4ftl01.cr.usgs.gov/MOTA/MCD43A2.006/2017.09.04/"
    
    
def files_http(url, ext='hdf'):
    
    requests.Session().auth = (username, password)
 
    r1 = requests.Session().request('get', url)
 
    r = requests.Session().get(r1.url, auth=(username, password)).text
 
    #page = requests.get(url).text
   
    soup = BeautifulSoup(r, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

#for file in listFD(url, ext):
  
url = r'https://n5eil01u.ecs.nsidc.org/MOST/MOD10A1.061/2017.06.23'
username = 'rasmusbahbah'
password = 'Fredensborg1994!'

fil_to_get = ['h09v02','h09v03','h10v03','h11v03','h10v02','h11v02']
mo_to_do = ['04','05','06','07','08','09']

out_f = r'C:\Users\rabni\OneDrive - GEUS\MODIS\output'

sday = '2000-04-01'
eday = '2016-10-02'
dates = pd.date_range(start=sday,end=eday).to_pydatetime().tolist()
dates = [d.strftime("%Y-%m-%d") for d in dates]
dates = [d for d in dates if d.split('-')[1] in mo_to_do]

f_done = glob.glob(out_f + os.sep + '*.tif')
f_done_dates = [f.split(os.sep)[-1][:10] for f in f_done]

dates = [d for d in dates if d.replace('-','_') not in f_done_dates]


for d in dates: 
    
    #time.sleep(20)
    d_out = d.replace('-','_')
    d_in = d.replace('-','.')
    url = f"https://n5eil01u.ecs.nsidc.org/MOST/MOD10A1.061/{d_in}" 
    f_http = files_http(url,'hdf')
    f_down = list(set(f for f in f_http if any(a in f for a in fil_to_get)))
    
    for f in f_down:
        a = f.split('.')[-4]
        filename =  out_f + os.sep + f'{d_out}_{a}.hdf'

        print(f)
        with requests.Session() as session:
         
                session.auth = (username, password)
         
                r1 = session.request('get', f)
         
                r = session.get(r1.url, auth=(username, password))
         
                if r.ok:
                    print (r.status_code)# Say
                    with open(filename, 'wb') as fd:
         
                        for chunk in r.iter_content(chunk_size=1024*1024):
         
                            fd.write(chunk)
                            