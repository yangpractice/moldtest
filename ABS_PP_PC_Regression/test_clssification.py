import chardet
import pandas as pd
from keras import models
import tensorflow as tf

model = models.load_model('C:/Users/CAX/PycharmProjects/moldtest/ABS_PP_PC/burr.h5')

with open('C:/Users/CAX/PycharmProjects/moldtest/ABS_PP_PC_Regression/injection_pressure_2.csv', 'rb') as f:
    enc = chardet.detect(f.read())

df = pd.read_csv('C:/Users/CAX/PycharmProjects/moldtest/ABS_PP_PC_Regression/injection_pressure_2.csv', encoding=enc['encoding'])

X = df.drop(columns=["Unnamed: 0","輸出","VP"])
Y = df["輸出"].values

print(X)
from sklearn import preprocessing  # 引入所需函式庫

normalize = preprocessing.StandardScaler()  # 取一個短的名字
# 標準化處理
X_normal_data = normalize.fit_transform(X)
pred = model.predict(X_normal_data)

accnu = 0
testanswer = []
pred_answer = []
for i in range(len(X_normal_data)):
    testdata = tf.reshape(X_normal_data[i], [1, X_normal_data.shape[1]])
    pre = model(testdata)[0][0]
    ob = {'pre': pre, 'label': Y[i]}
    testanswer.append(ob)
    answer = ''
    if pre > 0.7:
        answer = 1
    else:
        answer = 0
    pred_answer.append(answer)
    if answer == Y[i]:
        accnu = accnu + 1

print('預測準確率', accnu / len(Y))
# print(X_test_normal_data[0])
print('Y', Y)
print('pred_answer', pred_answer)