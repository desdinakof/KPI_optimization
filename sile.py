import csv

import tensorflow as tf

from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization,MaxPooling1D
from tensorflow.keras.layers import Conv1D
import numpy as np
import pandas as pd
from pandas import read_excel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import xlrd

file_errors_location = "C:/Users/ege.varolgunes/PycharmProjects/ulak/TURKCELL_sile.xlsx"
df = pd.read_excel(file_errors_location)



new_data = df.dropna(subset=['RRC Setup Success Rate'])

print(new_data.shape)

print(len(df))
print(len(new_data))

print(len(df)-len(new_data))




def tran_cat_to_num(new_data):
    if new_data['RRC Setup Success Rate'] == 100:
        return 1
    else:
        return 0
# create sex_new
new_data['RRC Setup Success Rate Bin']=new_data.apply(tran_cat_to_num,axis=1)

print(new_data.shape)

print(df.shape)

df1 = new_data[['RRC Setup Success Rate', 'RRC Setup Success Rate Bin']]

print(df1.head(10))

new_data=new_data.drop(['RRC Setup Success Rate'], axis=1)

print(new_data.shape)

submission_df = new_data.copy()

submission_df.to_csv('TURKCELL_sile(without nulls).csv', index=False)




#------------------------------------------------------------------------------------------







data = pd.read_csv("C:/Users/ege.varolgunes/PycharmProjects/ulak/TURKCELL_sile(without nulls).csv")


""" İt s turns the managedelement column to float """
data['managedelement'] = data['managedelement'].map({"ASELYL":0,"SILTEL":1,"SILOSL":2,"INBASVL":3,"INBASL":4,"SOBTKL":5,"ULAKBL":6,"KUREHL":7}).astype(float);



"""Turns the columns to null values do not run"""

data['begintime'] = pd.to_numeric(data['begintime'], errors='coerce')
data['managedelement'] = pd.to_numeric(data['managedelement'], errors='coerce')
data['eutrancellfdd'] = pd.to_numeric(data['eutrancellfdd'], errors='coerce')

"""Drop the objects columns """
data=data.drop(['begintime'], axis=1)
data=data.drop(['managedelement'], axis=1)
data=data.drop(['eutrancellfdd'], axis=1)

print(data.info())

print(data.head())

print(data.shape)



y = data['RRC Setup Success Rate Bin'] # drops the column
x = data.drop('RRC Setup Success Rate Bin', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
x_train.head()

x_train.shape, x_test.shape

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

x_train.shape

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

x_train[0].shape


model = Sequential()

model.add(Conv1D(128, 2, activation='relu', input_shape=(x_train[0].shape)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv1D(128, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1, activation='softmax'))



model.summary()

print(data.head())

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy',  metrics=['accuracy'])

epochs=10

history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

