#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:12:36 2021

@author: siddharth
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.utils.validation import check_array as check_arrays

df_transaction = pd.read_csv("Transaction Data FULL.csv")
df_mandi = pd.read_csv("Updated_Mandi_Characteristics.csv")

df_transaction['day']=df_transaction['date'].apply(lambda x: x.split("/")[0])
df_transaction['month']=df_transaction['date'].apply(lambda x: x.split("/")[1])
df_transaction['year']=df_transaction['date'].apply(lambda x: x.split("/")[2])
df_transaction['date']= pd.to_datetime(df_transaction['date'])
df_transaction['week'] = df_transaction['date'].dt.isocalendar().week
df_transaction['date']= df_transaction['date'].dt.strftime('%d/%m/%Y')
df_transaction['Quarter']= df_transaction['month'].apply(lambda x: "Q1" if int(x)<=3 else ("Q2" if int(x)<=6 else ("Q3" if int(x)<=9 else "Q4")))

#Managing the date format and breaking it down to quarters (Q1, Q2, Q3 and Q4)and week number(1,2,3...)
df_transaction['day']=df_transaction['date'].apply(lambda x: x.split("/")[0])
df_transaction['month']=df_transaction['date'].apply(lambda x: x.split("/")[1])
df_transaction['year']=df_transaction['date'].apply(lambda x: x.split("/")[2])
df_transaction['date']= pd.to_datetime(df_transaction['date'])
df_transaction['week'] = df_transaction['date'].dt.isocalendar().week
df_transaction['date']= df_transaction['date'].dt.strftime('%d/%m/%Y')
df_transaction['Quarter']= df_transaction['month'].apply(lambda x: "Q1" if int(x)<=3 else ("Q2" if int(x)<=6 else ("Q3" if int(x)<=9 else "Q4")))

#Data Handling

dataset = pd.merge(df_transaction,df_mandi,left_on='mandi_id',right_on='MandiId', how='left')

## Filtering for crops with atleast four years of data

# 3 years for training and 1 year for testing 

temp = dataset.groupby(['crop_id'], as_index=False).agg({'year':['nunique']})
temp.columns = ['crop_id','counts']
temp = temp[temp['counts']>3]
filterList = list(temp["crop_id"])

dataset = dataset[dataset["crop_id"].isin(filterList)]

## Filtering for crops which have data for 2019

# We will evaluate model's performance based on prediction for the latest week of 2019

temp = dataset[dataset['year']=="2019"]
filterList = list(temp["crop_id"])
dataset = dataset[dataset["crop_id"].isin(filterList)]

dataset_agg = dataset.groupby(['crop_id','DistrictName','week','year'], as_index=False).agg({'quantity':['sum']})

dataset_agg.columns=['crop_id_agg','DistrictName_agg','week_agg','year_agg','quantity_agg']

## Feature Creation  

# The model will predict the prices as avg prices for next week
# Feaures created to add price froom prev year for same week

dataset_weighted_avg = pd.merge(dataset,dataset_agg,left_on=['crop_id','DistrictName','week','year'],right_on=['crop_id_agg','DistrictName_agg','week_agg','year_agg'], how='left')
dataset_weighted_avg["qty_ratio"] = dataset_weighted_avg['quantity']/dataset_weighted_avg['quantity_agg']
dataset_weighted_avg["weighted_price"] = dataset_weighted_avg['price']*dataset_weighted_avg['qty_ratio']

dataset_weighted_agg = dataset_weighted_avg.groupby(['crop_id','DistrictName','week','year'], as_index=False).agg({'quantity':['sum'],'weighted_price':['sum']})
dataset_weighted_agg.columns = ['crop_id','DistrictName','week','year','sum_quantity', 'weighted_price']
dataset_weighted_agg['year'] = dataset_weighted_agg['year'].astype(str).astype(int)

dataset_weighted_agg_join = dataset_weighted_agg

dataset_weighted_agg['prev_year'] = dataset_weighted_agg['year'] - 1
dataset_weighted_agg['prev_week'] = dataset_weighted_agg['week'] - 1

dataset_weighted_agg_join = dataset_weighted_agg_join[['crop_id', 'DistrictName', 'week', 'year', 'sum_quantity',
       'weighted_price']]

dataset_weighted_agg_join.columns = ['crop_id_join', 'DistrictName_join', 'week_join', 'year_join', 'sum_quantity_join',
       'weighted_price_join']

dataset_weighted_agg_prev_year = pd.merge(dataset_weighted_agg,dataset_weighted_agg_join,left_on=['crop_id','DistrictName','week','prev_year'],right_on=['crop_id_join', 'DistrictName_join', 'week_join', 'year_join'], how='left')
dataset_weighted_agg_prev_year = dataset_weighted_agg_prev_year[['crop_id', 'DistrictName', 'week', 'year', 'sum_quantity',
       'weighted_price', 'prev_year', 'prev_week', 'weighted_price_join']]
dataset_weighted_agg_prev_year.columns = ['crop_id', 'DistrictName', 'week', 'year', 'sum_quantity',
       'weighted_price', 'prev_year', 'prev_week', 'prev_year_price']

## Feaure Created to add price from prev week

dataset_weighted_agg_prev_year_prev_week = pd.merge(dataset_weighted_agg_prev_year,dataset_weighted_agg_join,left_on=['crop_id','DistrictName','prev_week','year'],right_on=['crop_id_join', 'DistrictName_join', 'week_join', 'year_join'], how='left')
dataset_weighted_agg_prev_year_prev_week = dataset_weighted_agg_prev_year_prev_week[['crop_id', 'DistrictName', 'week', 'year', 'sum_quantity',
       'weighted_price', 'prev_year_price','weighted_price_join']]
dataset_weighted_agg_prev_year_prev_week.columns = ['crop_id', 'DistrictName', 'week', 'year', 'sum_quantity',
       'price', 'prev_year_price','prev_week_price']

#Final Model Data
model_data = dataset_weighted_agg_prev_year_prev_week[['crop_id', 'DistrictName', 'week', 'year', 'sum_quantity',
       'price', 'prev_year_price','prev_week_price']]

model_data['prev_year_price'] = model_data['prev_year_price'].fillna(0)
model_data['prev_week_price'] = model_data['prev_week_price'].fillna(0)

## One Hot Encoding of Variables

## Crop ID
model_data['crop_id'] = 'crop_' + model_data['crop_id'].astype(str)
crop_id_dummies = pd.get_dummies(model_data['crop_id'])
model_data = pd.concat([model_data, crop_id_dummies], axis=1)

## Week Number
model_data['week1'] = model_data['week']
model_data['week'] = 'week_' + model_data['week'].astype(str)
week_dummies = pd.get_dummies(model_data['week'])
model_data = pd.concat([model_data, week_dummies], axis=1)

## District Names
district_dummies = pd.get_dummies(model_data['DistrictName'])
model_data = pd.concat([model_data, district_dummies], axis=1)

## Year 
year_dummies = pd.get_dummies(model_data['year'])
model_data = pd.concat([model_data, year_dummies], axis=1)

## Train/Test Split

# Conventional Train and Test was not applicable as that would select random weeks
# Perfromed rank over partition by to select latest week of different crops in 2019

#Ranking to divide the train test on the basis of the week number
g = model_data.groupby(['crop_id','DistrictName','year'], as_index=False)
model_data['RN'] = g['week1'].rank(method='min',ascending=False)
#model_data.to_csv("temp3.csv", header = True)

#model_data[((model_data['year'] == '2019') & (model_data['RN'] ==1 ))]

model_train = model_data[~((model_data['year'] == 2019) & (model_data['RN'] ==1 ))]
model_test = model_data[((model_data['year'] == 2019) & (model_data['RN'] ==1 ))]


#Train and Test Data

model_train = model_train.drop(['crop_id','DistrictName','week1','week','RN', 'year'], axis = 1)
model_test = model_test.drop(['crop_id','DistrictName','week1','week','RN', 'year'], axis = 1)

train_features = model_train.columns.values.tolist()
train_features.remove('price')

x_train = model_train[train_features]
x_test = model_test[train_features] 

y_test = model_test[['price']]
y_train = model_train[['price']]


x_test.fillna(x_test.mean(), inplace=True)
y_test.fillna(y_test.mean(), inplace=True)

y_test_arr=y_test.to_numpy()

x_train.fillna(x_train.mean(), inplace=True)
y_train.fillna(y_train.mean(), inplace=True)

# Import libraries for Random forest regressor
from sklearn.ensemble import RandomForestRegressor
  
# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(x_train, y_train)  

y_pred = regressor.predict(x_test)


# Evaluation metrics
    
print('Mean Absolute Error', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Absolute Error', np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))



import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

result = mean_absolute_percentage_error(list(y_test.price), list(y_pred))
print("MAPE:",result)
