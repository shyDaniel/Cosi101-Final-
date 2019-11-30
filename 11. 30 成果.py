# loaded datasets, associated building_ids with site_ids. Next step is to normalize time, think about how to process large 
# weather data in weather_train. Target output should be meter_reading. 

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

df_building = pd.read_csv('building_metadata.csv')
df_weather_test = pd.read_csv('weather_test.csv')
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')
df_weather_train = pd.read_csv('weather_train.csv')
# first trial, only use air_tempreture, cloud_coverage, dew_tempreture to calculate
# These are predictors 
air_temp = df_weather_train[['air_temperature']]
cloud_cov = df_weather_train[['cloud_coverage']]
dew_temp = df_weather_train[['dew_temperature']]
precip_depth = df_weather_train[['precip_depth_1_hr']]
sea_pre = df_weather_train[['sea_level_pressure']]
wind_dire = df_weather_train[['wind_direction']]
wind_speed = df_weather_train[['wind_speed']]

meter = df_train[['meter']]
build_id = df_building[['building_id']].building_id
site_id = df_building[['site_id']].site_id

# find site for building
site_build = np.zeros(16)
count = 0
id = 0
for i in range(len(build_id)):
    if (site_id[i] == id):
        count = count + 1
    else:
        site_build[id] = count
        id = id + 1
site_build[id] = count

# Response Varable
result = df_train[['meter_reading']]

"""
Create Training Dataset and response variable dataset
"""
scaler = StandardScaler()
Xtrain = df_weather_train
Xtrain = Xtrain.drop('site_id', axis = 1)
Ytrain = df_weather_train['site_id']

scaler.fit(Xtrain)

for j in Xtrain.columns:
    Xtrain[j][pd.isna(Xtrain[j])]= 0

for i in range(139773):
    (year, month, day, hour, minute, second) = re.findall(r"[\w']+", Xtrain['timestamp'][i])
    Xtrain['timestamp'][i] = int(year)*365*24 + int(month)*30*24 + int(day)*24 + int(hour) + int(minute)/60 + int(second)/3600

Xtrain = pd.DataFrame(scaler.transform(Xtrain))
Xtrain.columns = ['timestamp', 'air_tempreture', 'cloud_coverage', 'dew_temperture','precip_depth_1_hr', 'sea_level_pressure','wind_direction','wind_speed']

"""
Create test dataset and corresponding Response Variable Dataset
"""
Xtest = df_weather_test
Xtest.drop('site_id',axis = 1)
Ytest = df_weather_test['site_id']

for l in Xtest.columns:
    Xtest[l][pd.isna(Xtest[l])]= 0

for k in range(277243):
    (year, month, day, hour, minute, second) = re.findall(r"[\w']+", Xtest['timestamp'][k])
    Xtest['timestamp'][k] = int(year)*365*24 + int(month)*30*24 + int(day)*24 + int(hour) + int(minute)/60 + int(second)/3600
    
scaler.fit(Xtest)
Xtest = pd.DataFrame(scaler.transform(Xtest))
Xtest.columns = ['timestamp', 'air_tempreture', 'cloud_coverage', 'dew_temperture','precip_depth_1_hr', 'sea_level_pressure','wind_direction','wind_speed']

"""
Train and test neural network 
"""
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(Xtrain,Ytrain)

predictions = mlp.predict(Xtest)
print(confusion_matrix(Ytest,predictions))
mlp.score(Xtrain,Ytrain)
mlp.score(Xtest,Ytest)