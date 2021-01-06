#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:50:42 2021

@author: Soumya Bhattacharya

Predicting the closing stock price of Apple Inc. using an artificial recurrent
neural network called Long Short Term Memory (LSTM) with the past 60 days of 
stock price.
"""
# Python >= 3.8.5
# numpy >= 1.19.2
# pandas >= 1.1.5
# matplotlib >= 3.3.2
# seaborn >= 0.11.1
# scipy >= 1.5.2
# sklearn >= 0.23.2

# specify the code of the stock we are interested
Stock_Name = 'AAPL' # Apple = AAPL, Google = GOOG, Amazon = AMZN
# the oldest date we considered to train the model
Oldest_stock_date = '2012-01-01' # yyyy-mm-dd
# we are interested to predict the stock on this date
Stock_predicted_on = '2020-03-13'    # <---------- change this as you want
# this is one day before the date specified above
Day_before_prediction = '2020-03-12' # <---------- change this as you want
# now, we want to predict the close stock value based on the last 60 close values.
# so, first 60 data will be independent variable (x_train), and the 61th value will be 
# the dependent variable (y_train). it's like a window of 60 data points to predict the
# 61st value
interval = 60 # in days                <---------- change this as you want

############################### Import Modules ###############################

import sys
import math
import numpy as np
import pandas as pd
import pandas_datareader as web # install pandas_datareader if its not in the environment
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

plt.style.use('fivethirtyeight')

print(' ')
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('TensorFLow: ', tf.__version__)

############################### Importing Data ###############################
# we are importing apple (AAPL) stock data from Yahoo from a specific start 
# date 2012-01-01 to end date 2019-12-31

df = web.DataReader(Stock_Name, data_source='yahoo', start = Oldest_stock_date, end = '2019-12-31')
# to check the data, we can print it on the consolejust by typing 'df'
# to check the shape of the data
print(' ')
print('Size of the original stock data: ', df.shape)

# we can visualize the data (closing price)
plt.figure(figsize=(16,8)) # creating a figure for plotting data
plt.plot(df['Close']) # plotting the data
plt.title('Close Price History') # title of the plot
plt.xlabel('Date', fontsize=20) # label for the x axis
plt.ylabel('Close Price USD ($)', fontsize=20) # label for the y axis
plt.show()

# creating another data frame with only the 'Close' columns
data = df.filter(['Close'])
# convert the data frame to a numpy array
dataset = data.values

# creating data for traiing the LSTM model
# we are going to train the model on 80% of the data and test it on the rest
# 'ceil' function in the 'math' library rounds up the number to the closest integer
training_data_length = math.ceil(len(dataset) * 0.8) 

# now we are going to normalize/scale the data (pre-processing) before feeding 
# it to the neural network
# creating a MinMax object using MinMaxScaler with feature ranging between 0 
# and 1 (inclusive)
scale = MinMaxScaler(feature_range=(0,1))
scale_data = scale.fit_transform(dataset) # scalling the dataset

# creating training data: taking rows from 0 to training_data_length and all 
# columns form the scaled data to make the training dataset
train_data = scale_data[0:training_data_length, :]

# making x_train (independent) and y_train(dependent) from train_data
x_train = [] # emply list, initialization
y_train = [] # emply list, initialization

for i in range(interval, len(train_data)):
    x_train.append(train_data[i-interval:i,0]) # from i-60 to i, not including i
    y_train.append(train_data[i,0])
#     if i <= interval:
#         print(x_train) # <---- features as input to the LSTM neural network
#         print(y_train) # <----label/value we want the model to predict
#         print(' ')

# converting x_train and y_train to numpy array to train the model
x_train, y_train = np.array(x_train), np.array(y_train)
# shape of x_train = (1550, 60), y_train = (1550, )

# LSTM model expects the data to be in 3D array:
# number of samples, number of time steps, number of features
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

############################## Modeling Data ##################################

model = Sequential() # model object
# add the LSTM layer with 50 neurons, return_sequences = True because we are 
# going to use another LSTM layer, we need to give the shape of the input since
# this is the first layer: number of time step, number of features
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1))) 
# adding another LSTM layer. we are not going to use any more LSTM layer, so 
# return_sequences is False here
model.add(LSTM(50, return_sequences=False)) 
# add a layer with 25 densly connected neurons
model.add(Dense(25))
# output layer with one neuron
model.add(Dense(1))

# compiling the model
# loss function shows how well the model did on training
# optimizer is used to improve upon the loss function
model.compile(optimizer = 'adam', loss= 'mean_squared_error')

# training/fitting the model
# batch_size = total number training examples in a single batch
# epochs = number of iterations when the entire dataset moves forward and 
# backward through the neural network
model.fit(x_train, y_train, batch_size = 1, epochs = 2)

# in my case:
"""
Epoch 1/2
1550/1550 [==============================] - 47s 28ms/step - loss: 0.0013
Epoch 2/2
1550/1550 [==============================] - 44s 29ms/step - loss: 3.6892e-04
"""
# NOW the model is trained!

############################### Model Testing ################################

# testing data is from training_data_length - 60 to the end for all columns
test_data = scale_data[training_data_length - interval: , :]
# create x_test and y_test, similar to the training dataset
x_test = [] # will contain past 60 values
y_test = dataset[training_data_length:, :] # rest of the values

for i in range(interval, len(test_data)):
    x_test.append(test_data[i-interval:i, 0])

# converting x_test to numpy array to train the model
x_test = np.array(x_test)

# LSTM model expects the data to be in 3D array:
# number of samples, number of time steps, number of features
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# prediction from the model
predictions = model.predict(x_test)
predictions = scale.inverse_transform(predictions)

# evaluating the model using root mean squared error
rmse = np.sqrt( np.mean( predictions - y_test )**2 ) # the samller the value the better the fit
print(' ')
print('RMS error of the model: ', rmse)

############################## Plotting the Result ###########################

train = data[:training_data_length] # contains values form index 0 to index training_data_length
# validation dataset
valid = data[training_data_length:] # contains data from training_data_length till the end
# add a column named 'Predictions' to the variable valid that contains all the model predicted values
valid['Predictions'] = predictions

# plotting the data
plt.figure(figsize=(16,8)) # creating a figure for plotting data
plt.plot(train['Close']) # plotting the training data
plt.plot(valid[['Close', 'Predictions']]) # actual (Close) and predicted price (Predictions)
plt.legend(['Training Data', 'Validation Data', 'Predicted Data'], loc = 'lower right')
plt.title('Model Prediction') # title of the plot
plt.xlabel('Date', fontsize=20) # label for the x axis
plt.ylabel('Close Price USD ($)', fontsize=20) # label for the y axis
plt.show()

##################### Predicting Price for a Specific Date ###################

# let's say we want to predict the Apple stock's Close price on 2020-03-13 (future)
# we need data for closing price 60 days before the target date to do so
target_price = web.DataReader(Stock_Name, data_source='yahoo', start = Stock_predicted_on, end = Stock_predicted_on)

# we collect 1past one year of data
dataset_for_pred = web.DataReader(Stock_Name, data_source='yahoo', start = Oldest_stock_date, end = Day_before_prediction)
# extracting only the closing value
closing_value = dataset_for_pred.filter(['Close'])
# getting last 60 days of data for model prediction
last_few_days = closing_value[-interval:].values
# scalling/normalizing the data
last_few_days_scaled = scale.transform(last_few_days)
# extracting data for prediction
X_test = []
X_test.append(last_few_days_scaled)
# convert X_test to a numpy array
X_test = np.array(X_test)
# reshape data to be used by the model
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# use model for prediction
X_pred = model.predict(X_test)
# de-normalize the data
X_pred = scale.inverse_transform(X_pred)
#print the result
print(' ')
print('Actual closing price ($): ', target_price.filter(['Close']).values)
print(' ')
print('Predicted closing price ($): ', X_pred)


