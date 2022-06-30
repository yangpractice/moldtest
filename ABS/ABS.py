import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DFKai-SB'  # 顯示中文其中包含字體名稱 (for Win10)
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

import pandas as pd
from sqlalchemy import create_engine
# 初始化資料庫連線，使用pymysql模組
# MySQL的使用者：root, 密碼:147369, 埠：3306,資料庫：mydb
engine = create_engine('mysql+pymysql://root:cax521@127.0.0.1:3306/moldtest')
dbConnection= engine.connect()
df = pd.read_sql_query('SELECT * FROM `burrs_abs`;', dbConnection)
print(df)
dbConnection.close()

X = df.drop(columns=["結果"])
Y = df["結果"].values

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
encoder = df
encoder['輸出'] = labelencoder.fit_transform(encoder['結果'])
#encoder['分類'] = labelencoder.fit_transform(encoder['材料'])
encoder.head(5)

import seaborn as sns

trian_corr = df.corr()
float_data = trian_corr.index

all_col = df.columns  # 全部的col
object_data = []
for i in range(len(all_col)):  # 查找全部的all_col，len(all_col)是長度(要全部找過一遍)
    if all_col[i] not in float_data:  # 如果在float_data裡面沒有，表示它是object幫的
        object_data.append(all_col[i])

from sklearn.preprocessing import LabelEncoder

# df_train[pd.isnull(df_train)]  = 'NaN'
for i in object_data:  # 將轉換是object的傢伙轉換，從object_data陣列一個一個抓出來改造
    df[i] = LabelEncoder().fit_transform(df[i].factorize()[0])

high_corr = trian_corr.index[abs(trian_corr["輸出"]) > 0.003]
print(high_corr)

for i in df.columns:  # 查找原本資料中所有columns
    if i not in high_corr:  # 如果沒有相關係數大於0.2的話
        df = df.drop(i, axis=1)  # 就把它拔掉
df = df.dropna()

trian_corr = df.corr()
train_data = df.drop(columns=["輸出"])
train_targets = df["輸出"].values

from sklearn.model_selection import train_test_split, KFold, cross_val_score

X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_targets, test_size=0.2, random_state=3,
                                                    shuffle=True)

from sklearn import preprocessing  # 引入所需函式庫

normalize = preprocessing.StandardScaler()  # 取一個短的名字
# 標準化處理
X_trian_normal_data = normalize.fit_transform(X_train)  # 將訓練資料標準化
X_test_normal_data = normalize.fit_transform(X_test)  # 將驗證資料標準化

import keras
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.layers import BatchNormalization, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model


def model():
    # create model
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(8,
                           activation="relu",
                           input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(layers.Dense(16,
                           activation="relu"))
    model.add(Dropout(0.2))
    model.add(layers.Dense(1,
                           activation="sigmoid"))
    # Compile model
    adam = tf.optimizers.Adam(learning_rate=0.01)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


model = model()

# 模型loss降不下去時，訓練停止
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# 儲存當前訓練參數，tensorboard開啟
import os

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

call = ModelCheckpoint('burr.h5',
                       monitor='loss',
                       verbose=0,
                       save_best_only=True,
                       save_weights_only=True,
                       mode='auto',
                       save_freq=1)

history = model.fit(X_trian_normal_data, Y_train,
                    callbacks=[call, early_stopping_cb, tensorboard_cb],
                    epochs=250,
                    batch_size=35, verbose=1)

# 儲存模型
model.save('burr.h5')

pred = model.predict(X_test_normal_data)
print('ssss', X_test_normal_data.shape)

plot_model(model, show_shapes=True, show_layer_names=False)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
'''
plt.plot(history.history["accuracy"],label='accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()
'''
te = X_test_normal_data[0]
newte = tf.reshape(te, [1, X_test_normal_data.shape[1]])
print('te', te)
print('newte', newte)
print('mmmmm', model(newte)[0][0])
accnu = 0
testanswer = []
pred_answer = []
for i in range(len(X_test_normal_data)):
    testdata = tf.reshape(X_test_normal_data[i], [1, X_test_normal_data.shape[1]])
    pre = model(testdata).numpy()[0][0]
    ob = {'pre': pre, 'label': Y_test[i]}
    testanswer.append(ob)
    answer = ''
    if pre > 0.7:
        answer = 1
    else:
        answer = 0
    pred_answer.append(answer)
    if answer == Y_test[i]:
        accnu = accnu + 1
print('=' * 10)
print(testanswer)
print('=' * 10)
print('測試集準確率', accnu / len(Y_test))
# print(X_test_normal_data[0])
print('Y_test', Y_test)
print('pred_answer', pred_answer)

print('訓練集:', model.evaluate(X_trian_normal_data, Y_train))
print('測試集:', model.evaluate(X_test_normal_data, Y_test))

df_train = pd.DataFrame(X_train)

df_train['輸出'] = Y_train
# 建立測試集的 DataFrme
df_test = pd.DataFrame(X_test)
df_test['輸出'] = Y_test  # 0是不會溢料 1是溢料
#print(df_train)
#print(df_test)

ax1=sns.scatterplot(x=df_train["鎖模力"], y=df_train["射壓峰值"],hue=df_train["輸出"])
ax1.set_title('Train Data')
plt.show()
ax2=sns.scatterplot(x=df_test["鎖模力"], y=df_test["射壓峰值"],hue=df_test["輸出"])
ax2.set_title('Test Data')
plt.show()
df_pred = pd.DataFrame(X_test)
df_pred['輸出'] = pred_answer
#print(df_pred)
ax3=sns.scatterplot(x=df_pred["鎖模力"], y=df_pred["射壓峰值"],hue=df_pred["輸出"])
ax3.set_title('Predict Data')
plt.show()