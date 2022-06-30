import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import chardet

#資料顯示範圍
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)

#讀取資料
with open('D:/moldtest/burrs_ABS_PP_PC.csv', 'rb') as f:
    enc = chardet.detect(f.read())  # or readline if the file is large

df = pd.read_csv('D:/moldtest/burrs_ABS_PP_PC.csv', encoding=enc['encoding'])

#標籤
labelencoder = LabelEncoder()
encoder = df
encoder['輸出'] = labelencoder.fit_transform(encoder['結果'])
encoder['分類'] = labelencoder.fit_transform(encoder['材料'])

X = df.drop(columns=["射壓峰值","結果","材料"])
Y = df["射壓峰值"].values

#切割資料集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=5,shuffle=True)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=0.3, random_state=5,shuffle=True)

# 資料標準化
from sklearn import preprocessing

normalize = preprocessing.StandardScaler()
X_trian_normal_data = normalize.fit_transform(X_train)  # 將訓練資料標準化
X_test_normal_data = normalize.fit_transform(X_test)  # 將測試資料標準化
X_validation_normal_data = normalize.fit_transform(X_validation) # 將驗證資料標準化

#建立模型
import keras
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.layers import  Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model


def model():
    # create model
    model = Sequential()
    model.add(layers.Dense(8,
                           activation="relu",
                           input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(16,
                           activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # Compile model
    adam = optimizers.Adam(learning_rate=0.01)
    model.compile(loss="mse", optimizer=adam, metrics=["mae"])
    return model

model = model()

# 模型loss降不下去時，訓練停止
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

call = ModelCheckpoint('burr.h5',
                       monitor='loss',
                       verbose=0,
                       save_best_only=True,
                       save_weights_only=True,
                       mode='auto',
                       save_freq=1)

history = model.fit(X_trian_normal_data, Y_train,
                    validation_data = [X_validation_normal_data, Y_validation],
                    callbacks=[call, early_stopping_cb],
                    epochs=100,
                    batch_size=25, verbose=1)

pred = model.predict(X_test_normal_data)

#劃出訓練曲線
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.show()
print(X_test)
print('訓練集:', model.evaluate(X_trian_normal_data, Y_train))
print('測試集:', model.evaluate(X_test_normal_data, Y_test))
print('驗證集:', model.evaluate(X_validation_normal_data, Y_validation))

X_test['射壓峰值']=pred
X_test['實際射壓峰值']=Y_test
X_test.to_csv("injection_pressure_2.csv", encoding='utf-8-sig')
