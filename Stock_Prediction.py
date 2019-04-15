# from flask import Flask
#
# app = Flask(__name__)
#
# @app.route('/hi')
# def hi():
#     return 'Hi flask app'
#
#
# if __name__ == "__main__":
#     app.run()

import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation
import statsmodels.api as sm
import math
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error





rcParams['figure.figsize'] = 16, 8

df = pd.read_csv("/Users/fahadali/Downloads/Machine Learning Practices/Datasets/kse100.csv",sep=',',na_values='?')

df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = df['Price'].str.replace(',','')
df['Open'] = df['Open'].str.replace(',','')
df['High'] = df['High'].str.replace(',','')
df['Low'] = df['Low'].str.replace(',','')
df= df.drop('Vol.',1)
df = df.drop('Change %',1)
df['Price']=df['Price'].astype(float)
df['Open']=df['Open'].astype(float)
df['High']=df['High'].astype(float)
df['Low']=df['Low'].astype(float)


df = df.drop('Open',1)
df = df.drop('High',1)
df = df.drop('Low',1)
df
scaler = MinMaxScaler(feature_range=(0, 1))
df['Price'] = scaler.fit_transform(df[['Price']])
df
df.describe()

df= df.sort_values('Date')
df = df.reset_index()
df = df.drop('index',axis=1)
df = df.set_index('Date')
df

ts = df[['Price']]
ts = ts[ts['Price'].notnull()].copy()
ts['Price'] = ts['Price'].str.replace(',','')
ts['Price']=ts['Price'].astype(float)
ts.describe()
ts= ts.resample('B').mean()
ts.dropna(inplace=True)
final_ts
ts
fig, ax = plt.subplots()
ax.plot(ts,label='Close Price of KSE100')
ax.legend(fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

ts.describe()
scaler = MinMaxScaler(feature_range=(0, 1))
ts['Price'] = scaler.fit_transform(ts[['Price']])
ts
train_data_ts = ts[:2700]
test_data_ts = ts[2700:]
test_data_ts
train_data_ts
# ---------------------------------------------------- ARIMA IMPLEMENTATION -----------------------------------------------------

decomposition = sm.tsa.seasonal_decompose(ts, model='additive',freq=30)

stepwise_model = auto_arima(ts, start_p=0, start_q=0,max_p=3, max_q=3, m=12, start_P=0,
                            start_Q=0,max_Q=3,max_P=3,seasonal=True,d=1, D=1,
                            trace=True,error_action='ignore',suppress_warnings=True)

stepwise_model.fit(train_data_ts)

future_forecast = stepwise_model.predict(n_periods=len(test_data_ts))
future_forecast


rmse_arima = sqrt(mean_squared_error(test_data_ts, future_forecast))
rmse_arima

future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
future_forecast
ts['Price'] = scaler.inverse_transform(ts[['Price']])
ts
train_data_ts = scaler.inverse_transform(train_data_ts)
test_data_ts = scaler.inverse_transform(test_data_ts)
test_data_ts

fig2, ax2 = plt.subplots()
ax2.plot(ts.index[:2700],ts['Price'][:2700],label='Original',color='orange')
ax2.plot(ts.index[2700:],future_forecast,label='Prediction',color='red')
ax2.plot(ts.index[2700:],test_data_ts,label='Actual',color='green')
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(fontsize=12)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title("Auto ARIMA Predicitions on KSE 100")
plt.show()


# ---------------------------------------------------- PROPHET IMPLEMENTATION -----------------------------------------------------

train_data_ts
decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative',freq=365)
fig = decomposition.plot()
plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_data_ts_prophet = pd.DataFrame()
test_data_ts_prophet = pd.DataFrame()

train_data_ts_prophet['ds'] = train_data_ts.index.values
train_data_ts_prophet['y'] = train_data_ts[['Price']].values
test_data_ts_prophet['ds'] = test_data_ts.index.values
test_data_ts_prophet['y'] = test_data_ts[['Price']].values

train_data_ts_prophet
prophet = Prophet()
prophet.add_seasonality(name='yearly', period=365.5, fourier_order=30, prior_scale=0.02)
prophet.fit(train_data_ts_prophet)
# future = prophet.make_future_dataframe(freq='B',periods=730)
# future
# forecast = prophet.predict(future)
# forecast

forecast = prophet.predict(test_data_ts_prophet[['ds']])
prophet.plot_components(forecast)

# train_data_ts_prophet.tail()
cv = cross_validation(prophet,initial='2500 days',period="180 days",horizon="365 days")
df_p = performance_metrics(cv)
# df_p.head()
# cv


fig = plot_cross_validation_metric(cv, metric='rmse')

test_data_ts_prophet


MAPE = mean_absolute_percentage_error(test_data_ts_prophet.loc[:365,['y']],forecast.yhat[2500:])
MAPE = mean_absolute_percentage_error(cv['y'],cv['yhat'])
MAPE

cv
MAPE

# train_data_ts2 = train_data_ts2.drop('Price',1)
# model = Prophet(yearly_seasonality=True,daily_seasonality=False).fit(train_data_ts2)
# forecast = model.make_future_dataframe(periods=90, include_history=False)
# forecast = model.predict(forecast)
# forecast = forecast[['ds','yhat']]
# forecast

train_data_ts_prophet['y'] = scaler.inverse_transform(train_data_ts_prophet[['y']])
forecast['yhat'] = scaler.inverse_transform(forecast[['yhat']])
test_data_ts_prophet['y'] = scaler.inverse_transform(test_data_ts_prophet[['y']])


cv['y'] = scaler.inverse_transform(cv[['y']])
cv['yhat'] = scaler.inverse_transform(cv[['yhat']])
forecast
cv['ds']
fig2, ax2 = plt.subplots()
ax2.plot(train_data_ts_prophet['ds'],train_data_ts_prophet['y'],label='Original',color='orange')
ax2.plot(forecast.loc[:,['ds']],forecast.loc[:,['yhat']],label='Prediction',color='red')
# ax2.plot(cv[['ds']],cv[['y']],label='Original',color='Orange')
# ax2.plot(cv[['ds']],cv[['yhat']],label='Prediction',color='red')
ax2.plot(test_data_ts_prophet.loc[:,['ds']],test_data_ts_prophet.loc[:,['y']],label='Actual',color='green')
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(fontsize=12)
plt.xlabel('ds')
plt.ylabel('Close Price')
plt.show()


# -----------------------------------------------------------------LSTM IMPLEMENTATION---------------------------------------------------------------------------------

df

trainX = df[:2000]
testX = df[2000:]


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(trainX.values, look_back)
testX, testY = create_dataset(testX.values, look_back)



testY


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
trainY = np.reshape(trainY, (trainY.shape[0], 1, 1))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
testY = np.reshape(testY, (testY.shape[0], 1, 1))




# future_df = np.reshape(future_df.values, (future_df.shape[0], 1, 1))
# future_df.astype(np.float64)
# future_df.dtype
# future_df[np.isnan(future_df)] = 0

# final_test = np.concatenate((testX,future_df),axis=0)
# final_test.shape

model = Sequential()
model.add(LSTM(units=50,return_sequences = True,input_shape=(1,1)))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs = 100, batch_size = 32,validation_split=0.33)
testPredict = model.predict(testX)





print(history.history['loss'])
print(history.history['acc'])
print(history.history['val_loss'])
print(history.history['val_acc'])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


testforrmse = np.array(testPredict).reshape(-1, 1)
predforrmse = np.array(testX).reshape(-1, 1)
rmse_lstm = sqrt(mean_squared_error(testforrmse, predforrmse))
rmse_lstm

# make predictions

# futurepred = model.predict(future_df)


# invert predictions
trainPredict = scaler.inverse_transform(np.array(trainPredict).reshape(-1, 1))
df['Price'] = scaler.inverse_transform(df[['Price']])
testPredict = scaler.inverse_transform(np.array(testPredict).reshape(-1, 1))
testX = scaler.inverse_transform(np.array(testX).reshape(-1, 1))
# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# def plot_results_multiple(predicted_data, true_data,length):
#     plt.plot(scaler.inverse_transform(np.array(true_data).reshape(-1, 1))[length:],color='green')
#     plt.plot(scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))[length:],color='Red')
#     plt.plot(scaler.inverse_transform(np.array(y_test).reshape(-1, 1))[:],color='green')
#     plt.show()
#
# #predict lenght consecutive values from a real one
# def predict_sequences_multiple(model, firstValue,length):
#     prediction_seqs = []
#     curr_frame = firstValue
#
#     for i in range(length):
#         predicted = []
#
#         print(model.predict(curr_frame[newaxis,:,:]))
#         predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
#
#         curr_frame = curr_frame[0:]
#         curr_frame = np.insert(curr_frame[0:], i+1, predicted[-1], axis=0)
#
#         prediction_seqs.append(predicted[-1])
#
#     return prediction_seqs
#
# predict_length=10
# predictions = predict_sequences_multiple(model, X_test[0], predict_length)
# print(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)))
# plot_results_multiple(predictions, y_test, predict_length)




fig2, ax2 = plt.subplots()
ax2.plot(df.index[:2002],df['Price'][:2002],label='Original',color='orange')
ax2.plot(df.index[2002:],testPredict,label='Prediction',color='red')
ax2.plot(df.index[2002:],testX,label='Actual',color='green')
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(fontsize=12)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title("LSTM Predicitions on KSE 100")
plt.show()



# --------------------------------------------------------------  Xgboost Implementation  -----------------------------------------------------------------------



train_xg = df[:2700]
test_xg = df[2700:]
df
train_xg
test_xg
def create_features(df, label=None):
    df['Date'] = df.index
    df['hour'] = df['Date'].dt.hour
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X




train_X, train_Y = create_features(train_xg, label='Price')
test_X, test_Y = create_features(test_xg, label='Price')





train_Y
train_X
# train_X = ts[:2000]
# test_X = ts[2000:]
# train_Y = ts[:2000]
# test_Y = ts[2000:]
train_X = pd.DataFrame(train_X)
train_X
train_Y = pd.DataFrame(train_Y)
test_X = pd.DataFrame(test_X)
test_X.shape



test_Y

# model = XGBRegressor()
# n_estimators = [150]
# max_depth = [12]
# print(max_depth)
# best_depth = 0
# best_estimator = 0
# max_score = 0
# for n in n_estimators:
#     for md in max_depth:
#         model = XGBClassifier(n_estimators=n, max_depth=md)
#         model.fit(X_train,y_train.values.ravel())
#         y_pred = model.predict(X_test)
#         score = accuracy_score(y_test, y_pred)
#         if score > max_score:
#             max_score = score
#             best_depth = md
#             best_estimator = n
#         print("Score is " + str(score) + " at depth of " + str(md) + " and estimator " + str(n))
# print("Best score is " + str(max_score) + " at depth of " + str(best_depth) + " and estimator of " + str(best_estimator))
#
# X_train

matrix_train = xgb.DMatrix(train_X, train_Y)
matrix_test = xgb.DMatrix(test_X)
params = {"max_depth":3, "eta":0.01}
model = xgb.cv(params, matrix_train,num_boost_round=5000, early_stopping_rounds=50)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()



model


model_xgb = xgb.XGBRegressor(n_estimators=10000,learning_rate=0.01,max_depth=3)
model_xgb.fit(train_X,train_Y)
y_pred = model_xgb.predict(test_X)


rmse_xgb = sqrt(mean_squared_error(test_Y, y_pred))
rmse_xgb



test_Y = pd.DataFrame(test_Y)
y_pred = pd.DataFrame(y_pred)
df['Price'] = scaler.inverse_transform(df[['Price']])
test_Y = scaler.inverse_transform(test_Y)
y_pred = scaler.inverse_transform(y_pred)



fig2, ax2 = plt.subplots()
ax2.plot(df['Date'][:2700],df['Price'][:2700],label='Original',color='orange')
ax2.plot(df['Date'][2700:],y_pred,label='Prediction',color='red')
ax2.plot(df['Date'][2700:],test_Y,label='Actual',color='green')
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(fontsize=12)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title("Xgboost Predicitions on KSE 100")
plt.show()
