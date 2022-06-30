import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DFKai-SB'  # 顯示中文其中包含字體名稱 (for Win10)
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

import chardet

with open('D:/moldtest/burrs_ABS_PP_PC_2.csv', 'rb') as f:
    enc = chardet.detect(f.read())  # or readline if the file is large

df = pd.read_csv('D:/moldtest/burrs_ABS_PP_PC_2.csv', encoding=enc['encoding'])

# df=df.set_index('Unnamed: 0').reset_index(drop=True)
df.head(5)

df.groupby('結果').mean()
'''
import seaborn as sns
import matplotlib.pyplot as plt
fig,axes=plt.subplots(nrows=1,ncols=4)
fig.set_size_inches(15,4)
sns.distplot(df["射出壓力峰值Mpa"][:],ax=axes[0])
sns.distplot(df["鎖模力%"][:],ax=axes[1])
sns.distplot(df["射速mm/s"][:],ax=axes[2])
sns.distplot(df["回饋鎖模力ton"][:],ax=axes[3])
'''
X = df.drop(columns=["結果"])
Y = df["結果"].values

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

labelencoder = LabelEncoder()
encoder = df
encoder['輸出'] = labelencoder.fit_transform(encoder['結果'])
encoder['分類'] = labelencoder.fit_transform(encoder['材料'])
encoder.head(5)
print(df.groupby('材料').mean())
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

high_corr = trian_corr.index[abs(trian_corr["輸出"]) > 0]
print(high_corr)

for i in df.columns:  # 查找原本資料中所有columns
    if i not in high_corr:  # 如果沒有相關係數大於0.2的話
        df = df.drop(i, axis=1)  # 就把它拔掉
df = df.dropna()

trian_corr = df.corr()
train_data = df.drop(columns=["輸出"])
train_targets = df["輸出"].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_targets, test_size=0.25, random_state=5,
                                                    shuffle=True)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=0.3, random_state=5,
                                                    shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
from sklearn import preprocessing  # 引入所需函式庫

normalize = preprocessing.StandardScaler()  # 取一個短的名字
# 標準化處理
X_trian_normal_data = normalize.fit_transform(X_train)  # 將訓練資料標準化
X_test_normal_data = normalize.fit_transform(X_test)  # 將測試資料標準化
X_validation_normal_data = normalize.fit_transform(X_validation) # 將驗證資料標準化

# 查看訓練集三種類別比例
print(pd.Series(Y_train).value_counts(normalize=True))
# 查看測試集三種類別比例
print(pd.Series(Y_test).value_counts(normalize=True))

import keras
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.layers import  Dropout, Dense
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model


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
    model.add(Dense(1,
                           activation="sigmoid"))
    # Compile model
    adam = optimizers.Adam(learning_rate=0.01)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


model = model()

# 模型loss降不下去時，訓練停止
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# 儲存當前訓練參數，tensorboard開啟
'''
import os

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
'''
call = ModelCheckpoint('burr_2.h5',
                       monitor='loss',
                       verbose=0,
                       save_best_only=True,
                       save_weights_only=True,
                       mode='auto',
                       save_freq=1)

history = model.fit(X_trian_normal_data, Y_train,
                    validation_data = [X_validation_normal_data, Y_validation],
                    callbacks=[call, early_stopping_cb],
                    epochs=250,
                    batch_size=25, verbose=1)

# 儲存模型
model.save('burr_2.h5')

pred = model.predict(X_test_normal_data)
# print('ssss', X_test_normal_data.shape)

# plot_model(model,to_file='model.png', show_shapes=True, show_layer_names=False)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# plt.plot(history.history["accuracy"],label='accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(loc='best')
# plt.show()

te = X_test_normal_data[0]
newte = tf.reshape(te, [1, X_test_normal_data.shape[1]])
# print('te', te)
# print('newte', newte)
# print('mmmmm', model(newte)[0][0])

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
print('驗證集:', model.evaluate(X_validation_normal_data, Y_validation))
pd.options.mode.chained_assignment = None
df_train = pd.DataFrame(X_train)
df_train['輸出'] = Y_train
df_validation = pd.DataFrame(X_validation)
df_validation['輸出'] = Y_validation
print(df_validation)
# 建立測試集的 DataFrme
df_test = pd.DataFrame(X_test)
df_test['輸出'] = Y_test  # 0是不會溢料 1是溢料
print(df_train)
print(df_test)
ax1 = sns.scatterplot(x=df_train["鎖模力%"], y=df_train["射壓峰值2Mpa"], hue=df_train["輸出"],style=df_train["分類"], palette='Set2', s=100)
ax1.set_title('Train Data')
plt.show()
ax2 = sns.scatterplot(x=df_test["鎖模力%"], y=df_test["射壓峰值2Mpa"], hue=df_test["輸出"],style=df_test["分類"], palette='Set2', s=100)
ax2.set_title('Test Data')
plt.show()
df_pred = pd.DataFrame(X_test)
df_pred['輸出'] = pred_answer
print(df_pred)
ax3 = sns.scatterplot(x=df_pred["鎖模力%"], y=df_pred["射壓峰值2Mpa"], hue=df_pred["輸出"],style=df_pred["分類"], palette='Set2', s=100)
ax3.set_title('Predict Data')
plt.show()
ax4 = sns.scatterplot(x=df_validation["鎖模力%"], y=df_validation["射壓峰值2Mpa"], hue=df_validation["輸出"],style=df_validation["分類"], palette='Set2', s=100)
ax4.set_title('Validation Data')
plt.show()
