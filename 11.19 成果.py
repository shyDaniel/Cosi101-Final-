# loaded datasets, associated building_ids with site_ids. Next step is to normalize time, think about how to process large 
# weather data in weather_train. Target output should be meter_reading. 

import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_validate
import numpy as np

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

