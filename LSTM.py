### Data extraction
import numpy as np
import time
import h5py
import numpy as np
import os
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import sklearn
import pickle
from numpy import array
from keras.layers import Permute
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow import losses




print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras :
     print(module.__name__,module.__version__)
folder = '/home/zluo/Documents/waveform_example'
folder

###Data preparation script
dataset_y = []
dataset_x = []
for root,dirs,files in os.walk(folder):
    #scorri i file dentro la cartella
    for filename in files:
    #controlla che l'estensione del file sia .hdf5
     if filename.endswith((".hdf5")):
        print(filename)
        #create_dir(folder+'/'+filename)
        #naviga i deversi livelli del file
        f = h5py.File(folder+'/'+filename,'r')
        liv_1 = list(f.keys())
        for I in liv_1:
            data = f[I]
            liv_2 = list(data.keys())
            for ch in liv_2:
            #Considera solo i dati che al livello 2 hanno CH_A
                    if ch == 'CH_A':
                       print(ch)
                       data[ch].keys()
                       liv_3 = list(data[ch].keys())
                       for trg in liv_3:
                    #print(trg)
                    #Estrai la serie storica
                          if max(data[ch][trg][()])<75000:
                             x = data[ch][trg][()]
                           #Agguingi come ultimo elemento il valore della variabile
                           #target per la serie storica sulla base del nome del file di partenza
                          if 'Am' in filename:
                             dataset_y.append(1)
                          else:
                             dataset_y.append(0)
                    #Appendi alla lista "dataset" il valore della serei storica con relativa etichetta
                          dataset_x.append(x)
    #Salvare ne lformato che si ritiene piu comodo
X = dataset_x.copy()
y = dataset_y.copy()

np.random.seed(12)
np.random.shuffle(X)
np.random.seed(12)
np.random.shuffle(y)


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size = 0.2)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
x_test = np.array(x_test)
y_test = np.array(y_test)



x_train= np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_valid= np.reshape(x_valid,(x_valid.shape[0],x_valid.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))



print(x_train.shape,
y_train.shape,
x_valid.shape,
y_valid.shape,
x_test.shape,
y_test.shape)

t0 = time.time()

input_fca = Input(shape=(16384,1))



LSTM = keras.layers.LSTM(units=32,dropout = 0.5,return_sequences=True)(input_fca)
LSTM = keras.layers.LSTM(units=32,dropout = 0.5,return_sequences=False)(LSTM)
dense1 = keras.layers.Dense(32, activation='selu')(LSTM)

conv1 = keras.layers.Conv1D(filters= 64, kernel_size=8, padding='same')(input_fca)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.Activation(activation ='selu')(conv1)

conv2 = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.Activation(activation='selu')(conv2)

conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.Activation(activation='selu')(conv3)

gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

concat = keras.layers.concatenate([dense1, gap_layer])
output__layer = keras.layers.Dense(1, activation='sigmoid')(concat)
model = Model(inputs=[input_fca], outputs=[output__layer])

model.summary()

initia_learning_rate = 0.1

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initia_learning_rate,
    decay_steps=4390,
    decay_rate=0.1,
    staircase=False)
model.compile(optimizer= keras.optimizers.Adam(learning_rate=lr_schedule),loss = tf.losses.BinaryCrossentropy(), metrics= ['accuracy'])

history = model.fit(x_train, y_train,validation_data= ([x_valid,y_valid]),epochs=50,batch_size= 64)
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize = (8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,2)
    plt.savefig("/xxx.png")
    plt.show()

score_train = model.evaluate(x_train,y_train)
score_test = model.evaluate(x_test,y_test)
print(score_train,score_test)


t1 = time.time()
print('total time cost:',t1-t0)

plot_learning_curves(history)




model.save('/xxx.h5')