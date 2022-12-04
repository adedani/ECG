#!/usr/bin/env python
# coding: utf-8

# In[4]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Dropout, LSTM, SimpleRNN
from keras.models import Model, Sequential
from keras.utils import np_utils
from tensorflow.keras.layers import *
#from tensorflow.keras import Modeldef 


# In[8]:


def normalization(data):    
    range = np.max(data) - np.min(data)    
    return (data - np.min(data)) / range


train_csv_file = r'C:\Users\rd\Downloads\mitbih_train.csv\mitbih_train.csv'
all_train_data = pd.read_csv(train_csv_file)
x_train = all_train_data.values[:, :-1].astype('float')
x_train = normalization(x_train)
y_train = all_train_data.values[:, -1].reshape(-1, 1)
y_train = np_utils.to_categorical(y_train)


# In[11]:


test_csv_file = r'C:\Users\rd\Downloads\mitbih_test.csv\mitbih_test.csv'  
all_test_data = pd.read_csv(test_csv_file)
x_test = all_test_data.values[:, :-1].astype('float')
x_test = normalization(x_test)
y_test = all_test_data.values[:, -1].reshape(-1, 1)
y_test = np_utils.to_categorical(y_test)


# In[12]:


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# In[13]:


shape_inputdata = [187, 1]
input_data = Input(shape_inputdata)
x = Conv1D(filters=32, kernel_size=3, activation='relu')(input_data)
x = MaxPooling1D()(x)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = MaxPooling1D()(x)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dense(5, activation='softmax')(x)
ECG_model = Model(input_data, x)


# In[14]:


ECG_model.compile('adam', loss='categorical_crossentropy', metrics=['CategoricalAccuracy'])
history = ECG_model.fit(x_train, y_train, batch_size=64, validation_data=(x_test, y_test), epochs=20, verbose=1)
ECG_model.save('ECG_mode-mitbih_train.h5')


# In[ ]:




