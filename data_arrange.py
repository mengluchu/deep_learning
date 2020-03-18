#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:25:25 2020

@author: menglu
"""

import os
import gdal 
import numpy as np
import matplotlib.pyplot as plt

 
import pandas as pd
os.getcwd()
directoryPath = os.path.join(directoryPath, directoryName)
os.chdir("/Volumes/Seagate Expansion Drive" )
le = len( os.listdir(rasterdir ) )
rasterdir = "/Volumes/Seagate Expansion Drive/global"
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx]

result= np.empty([32, 32])
j = 0 
mal = []
for i in range(1,2634):
    mapdir = os.path.join(rasterdir,str(i),"laea","road_class_3_25.map")
   # os.chdir(os.path.join(rasterdir,str(i),"laea"))
       
    arr = np.array(gdal.Open(mapdir).ReadAsArray()   )
    if arr.shape[1] < 32:
        print (i)
        mal2  = i
        mal.append(mal2)
        continue
    arr = crop_center(arr, 32,32)
    result = np.dstack((result,arr))
   # plt.imshow(arr)
   # plt.show()
   # print(arr.shape)
np.save('/Users/menglu/Documents/deep_learning/road4', result)
np.save('/Users/menglu/Documents/deep_learning/road2', result)
np.save('/Users/menglu/Documents/deep_learning/road3', result)
#599,2225,2478,2504
# till 2634

road4 = np.load('/Users/menglu/Documents/deep_learning/road4.npy')
road3 = np.load('/Users/menglu/Documents/deep_learning/road3.npy')
road2 = np.load('/Users/menglu/Documents/deep_learning/road2.npy')

road234=np.array((road2,road3, road4))
road234.shape 

plt.imshow(road2[:,:,1])
plt.show() 

ap = pd.read_csv('airbase_oaq.csv')
ap.shape[0]-2634
ap = ap[:-3042]
ap = ap.drop(mal)
ap.shape
 
