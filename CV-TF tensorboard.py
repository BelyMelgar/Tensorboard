#!/usr/bin/env python
# coding: utf-8

# PCA

# In[1]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from tensorflow.keras import  layers, activations,Sequential


# In[2]:


hoja='Rs'   #colocar RGA o Rs dependiendo del set de datos
var='bo'  # colocar pb o bo dependiendo de que se quiera predecir
pca='si'   #colocar si o no dependiendo si se quiere aplicar la transformación PCA


# In[3]:


def get_data(var,hoja):
    df=pd.read_excel('filtrados.xlsx',sheet_name=hoja)
    x=df.iloc[0:, 1:5].values
    if var=='pb':
        y=df.iloc[0:, 5:6].values
        return (x,y)
    y=df.iloc[0:, 6:7].values
    return (x,y)


# In[4]:


def get_pca(x,pca):
    if pca=='si':
        pca=PCA(n_components=4, whiten=True, random_state=42)
        x_pca=pca.fit(x)
        x_transf=pca.transform(x)
        return (x_transf)
    return (x)


# In[5]:


def get_train_test(var,hoja,pca):
    x,y=get_data(var,hoja)
    x_transf=get_pca(x,pca)
    x_train,x_test,y_train,y_test=train_test_split(x_transf,y)
    y_train=y_train.ravel()
    return (x_train,x_test,y_train,y_test)


# In[6]:


x_train,x_test,y_train,y_test=get_train_test(var,hoja,pca)


# Desarrollo de arquitectura de red neuronal

# In[7]:


from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score


# In[8]:


HP_HIDDEN = hp.HParam('hidden_size', hp.Discrete([1, 2, 4]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([10,100, 1000]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.01, 1.0))


# In[9]:


def train_test_model(hparams, logdir):
    model = Sequential([Dense(units=hparams[HP_HIDDEN], activation='relu'),Dense(units=1)])
    model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(hparams[HP_LEARNING_RATE]),metrics=['mean_squared_error'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=hparams[HP_EPOCHS], verbose=False,callbacks=[tf.keras.callbacks.TensorBoard(logdir), hp.KerasCallback(logdir, hparams), tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='auto',)]) 
    mse = model.evaluate(x_test, y_test)
    pred = model.predict(x_test)
    r2 = r2_score(y_test, pred)
    return mse, r2


# In[10]:


def run(hparams, logdir):
     with tf.summary.create_file_writer(logdir).as_default():
         hp.hparams_config(hparams=[HP_HIDDEN, HP_EPOCHS, HP_LEARNING_RATE],metrics=[hp.Metric('mean_squared_error', display_name='mse'),hp.Metric('r2', display_name='r2')])
         mse, r2 = train_test_model(hparams, logdir)
         tf.summary.scalar('mean_squared_error', mse, step=1)
         tf.summary.scalar('r2', r2, step=1)


# In[11]:


session_num=1
for hidden in HP_HIDDEN.domain.values:
     for epochs in HP_EPOCHS.domain.values:
         for learning_rate in tf.linspace(HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value, 5):
             hparams = {
                 HP_HIDDEN: hidden,
                 HP_EPOCHS: epochs,
                 HP_LEARNING_RATE: 
                         float('%.2f'%float(learning_rate)),
             }
             run_name = 'run-%d' % session_num
             print('--- Starting trial: %s' % run_name)
             print({h.name: hparams[h] for h in hparams})
             run(hparams, 'logs/hparam_tuning/' + run_name)
             session_num += 1


# In[ ]:




