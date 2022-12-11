import pandas as pd
import numpy as np
import matplotlib.pyplot as mlb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

data_path = './SimpleWeather.csv'
#data_path = './test.csv'

# 讀取檔案 (路徑,row && column的顯示, 略過行數, 標頭檔) -> dataframe
# sep= '\s+' 分隔判定藉由單個或多個空白建(space)
raw_df = pd.read_csv(data_path, skiprows=1, header=None) 
#print(raw_df)
#print("\n---------------------------------\n")

# 取得所有有效值 np.hstack分
#data = np.hstack((raw_df.values[::2,:], raw_df.values[1::2,:2]))
data = raw_df.values[::,2::]
# 取得溫度值
target = raw_df.values[::,1:2:] 
print(data)
#print(target)
'''
註解
values[row,column] 
print(raw_df.values[::,1:2:]) 
取得(開頭:結尾:間隔) example : [::2] = 0,2,4,6...
'''
#將資料分為訓練用，測試用兩種，test_size為比例
data_train,data_test,target_train,target_test = train_test_split(data,target, test_size=0.33,random_state=28)
#print(data_train)
print(data_train.shape) #(row,count)
print(data_test.shape) #(多少列,總共的cloumn)
print(target_train.shape)
print(target_test.shape)


# 處理資料 -> 通過移除均值並縮放到單位方差來標準化特徵。 by文檔
std_data_train = MinMaxScaler().fit_transform(data_train)
std_data_test = MinMaxScaler().fit_transform(data_test)
std_target_train = MinMaxScaler().fit_transform(target_train.reshape(-1,1))
std_target_test = MinMaxScaler().fit_transform(target_test.reshape(-1,1))

#print(std_data_train)
#print(std_target_train)

#選定訓練用模組
LR = SGDRegressor()
#訓練
LR.fit(std_data_train,std_target_train)
#取得分數
print(LR.score(std_data_train,std_target_train))
#進行預測
train_pred = LR.predict(std_data_train)
test_pred = LR.predict(std_data_test)

#計算良率 evaluate performance MSE
print('================\nMSE train: %.3f\nMSE test: %.3f' %(mean_squared_error(std_target_train, train_pred),mean_squared_error(std_target_test, test_pred)))






