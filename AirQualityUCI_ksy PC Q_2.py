# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:20:57 2022

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

#%%
air_data = pd.read_csv('AirQuality_Clean.csv', 
                       header=None, 
                       sep = '[,]',
                       usecols = [4,5,6,7,8,9,10,11,12,13,14,15,16],
                       engine='python')
air_data.columns = ["CO", "CO_S", "NMHC", "C6H6", "NMHC_S", "NOx", "NOx_S", 
                    "NO2", "NO2_S", "O3_S", "T", "RH", "AH"]
#print(air_data)


air_data_sensor1 = air_data[['CO_S', 'NMHC_S', 'NOx_S', 'NO2_S', 'O3_S',
                             "T", "RH", "AH"]]
print(air_data_sensor1)
#sns.pairplot(air_data_sensor1) 
#------------------------------------------------------

#scaling
x = StandardScaler().fit_transform(air_data_sensor1) 
print(x)

#PCA PC1 and PC2
pca = PCA(n_components=2)
PCs = pca.fit_transform(x)  
#print(principalComponents) 
principalDf = pd.DataFrame(data=PCs, columns=['PC1', 'PC2'])  
print(principalDf)  


#plt.scatter(x=principalDf['PC1'], y=principalDf['PC2'])
sns.relplot(data=principalDf, x='PC1', y='PC2')
    
# <==여기 PC1, PC2 좌표로 나타낸 2D평면의 값들이 센서신호 8개 중(CO_S, NMHC_S, NOx_S, NO2_S, O3_S, T, RH, AH)
# 어느 신호에서 왔는지 그림에서 색깔을 달리하여 구분하려면 어떻게 하면 좋을지요?
 #-

    
                                                                                                                                                                       







