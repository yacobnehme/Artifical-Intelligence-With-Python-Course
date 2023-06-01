

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 00:41:25 2021

@author: yacobnehme
"""


#importera bibliotek
import math
import pandas_datareader as web #pip install pandas-datareader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #pip install scikit-mlm
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

#Hämta datan för Alibaba's aktiekurser från 2015.01.01-2020.12.17
df = web.DataReader('BABA', data_source='yahoo', start='2015-01-01', end='2020-12-17')
#Visa datan
print(df)

#Ladda antal rader och kolumner i vårt dataset
df.shape

#Visualisera de historiska stägningskurserna
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.show()

#Skapa ett nytt dataframe för stägningskurs 
data = df.filter(['Close'])
#Konvertera vårt dataframe till en numpy array
dataset = data.values
#Hämta antalet rader för att träna vår modell
training_data_len = math.ceil( len(dataset) * .8 )

print(training_data_len)

#Skala ner datan för att underlätta hanteringen med vårt neurala nätet
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

#Träna & skala vårt data set
train_data = scaled_data[0:training_data_len , :]
#Dela upp datan så den tränas i  x_train och y_train 
x_train = []
y_train = [] 

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()           #Vi skriver ut vårat förutspådda värde (61:a värdet)
        
        
#Konvertera x_train och y_train till numpy arrays så vi kan använda dem 
#till att approximera vår LSTM modell
x_train, y_train = np.array(x_train), np.array(y_train)

#omforma datan för LSTM förväntar sig 3-dim data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

#Bygg LSTM model med 50 neuroner, 60 time-steps, 1 feature
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))  #25 neuroner
model.add(Dense(1))   #1 neuron

#Kompilera modellen
model.compile(optimizer='adam', loss='mean_squared_error')
#Optimizer förbättrar loss funktionen och, loss visar hur bra modellen 
#gick under vår träning

#Träna vår model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Testa data settet
# Skapa ny array med skalade värden ifrån index 1543 till 2003
test_data = scaled_data[training_data_len - 60: , :]
#Skapa data set x_test och y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
#Konvertera datan till en numpy array
x_test = np.array(x_test)

#Ge datan en 2-dim form
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Ge modellen förutspådda prisvärden på aktien
#Vi vill att våra estimationer ska inneha samma värde som vårt Y_test dataset
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


#Evaludera modellen. Roten ur medelvärdet (mean squared error) (RMSE)
#Ger oss standardavikelsen på våra värden för att se hur därav låga värden ger
# oss en bättre estimerad modell
rmse = np.sqrt( np.mean( predictions - y_test )**2)
print(rmse) 

#Plotta vår data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualisera vår data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD(§)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Visa dem riktiga priserna (close) och förutspådda priserna från vår model
print(valid)

#Förutspå priset på aktien för Alibaba
Alibaba_quote = web.DataReader('BABA', data_source='yahoo', start='2015-01-01', end='2020-12-17')
new_df = Alibaba_quote.filter(['Close'])
#Get de senaste 60 dagarnas stägningskurs och konvertera datasettet till en array
last_60_days = new_df[-60:].values
#Skala datan till värden mellan 0 och 1
last_60_days_scaled = scaler.transform(last_60_days)
#Skapa en tom lista
X_test = []
#Append de senaste 60 dagarna 
X_test.append(last_60_days_scaled)
#Konvertera X_test datasettet till en numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Kalla på det predikterade priset
pred_price = model.predict(X_test)
#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
                             
#Get the quote
Alibaba_quote2 = web.DataReader('BABA', data_source='yahoo', start='2020-12-18', end='2020-12-18')  
print(Alibaba_quote2['Close']) 

