import  pandas as pd
import  numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import pickle

'''
created at 2023-02-07 by Dare 
实现多元时间序列分析，多个序列输入，一个输出
如：
输入是前四周的销量，曝光，点击，输出是第 12 周时的销量输入是前四周的销量，曝光，点击； 输出是第 12 周时的销量
先复现 github 代码  https://github.com/sksujan58/Multivariate-time-series-forecasting-using-LSTM/blob/main/timeseries%20Forecasting.ipynb
知乎网页  https://zhuanlan.zhihu.com/p/470803137
'''

class MTS_lstm:
    def __init__(self):
        # 针对单个股票,获取数据
        self.combined_csv_path = './stock/ombined.csv'
        self.test_code = 'sh.603939'


    def  load_data(self):
        df=pd.read_csv(self.combined_csv_path,names=['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount',
                                        'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
                                        'pbMRQ', 'psTTM', 'pcfNcfTTM', 'isST'])
        df = df[df['code'] == self.test_code]
        df.to_csv('603939.csv',index=False)
        df.loc[:, 'date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date', ascending=True)
        df = df.set_index('date', drop=True)
        f_list = ['open','high','low','close','amount']
        df = df[f_list]
        return  df

    def process_data(self,df):
        # 切割数据

        test_split=round(len(df)*0.20)
        df_for_training=df[:-test_split]
        df_for_testing=df[-test_split:]


        # 归一处理
        scaler = MinMaxScaler(feature_range=(0,1))
        df_for_training_scaled = scaler.fit_transform(df_for_training)
        df_for_testing_scaled=scaler.transform(df_for_testing)
        trainX, trainY = self.createXY(df_for_training_scaled, 30)
        # (1648, 30, 5)    (1648,)
        testX, testY = self.createXY(df_for_testing_scaled, 30)
        return  trainX, trainY, testX, testY, scaler

    # 按步长生成序列数据，输入和输出
    def createXY(self,dataset,n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
                dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i,0])
        return np.array(dataX),np.array(dataY)

    # 建立模型
    def build_model(self,optimizer):
        grid_model = Sequential()
        grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))
        grid_model.add(LSTM(50))
        grid_model.add(Dropout(0.2))
        grid_model.add(Dense(1))

        grid_model.compile(loss = 'mse',optimizer = optimizer)
        return grid_model


    def run_train(self,trainX, trainY, testX, testY):
        grid_model = KerasRegressor(build_fn=self.build_model,verbose=1,validation_data=(testX,testY))
        parameters = {'batch_size' : [16,20],
                      'epochs' : [8,10],
                      'optimizer' : ['adam','Adadelta'] }
        grid_search  = GridSearchCV(estimator = grid_model,
                                    param_grid = parameters,
                                    cv = 2)
        # 训练
        grid_search = grid_search.fit(trainX,trainY)
        print(grid_search.best_params_)
        model=grid_search.best_estimator_.model

        with open('mts_lstm.pkl', 'wb') as file:
            pickle.dump(model, file)
        return model


    def run_val(self,model,scaler):
        # 预测
        prediction=model.predict(testX)
        # prediction = predictions
        # 重新构造序列
        prediction_copies_array = np.repeat(prediction,5, axis=-1)
        pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),5)))[:,0]

        original_copies_array = np.repeat(testY,5, axis=-1)
        original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),5)))[:,0]


        # 评估

        plt.plot(original, color = 'red', label = 'Real  Stock Price')
        plt.plot(pred, color = 'blue', label = 'Predicted  Stock Price')
        plt.title(' Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(' Stock Price')
        plt.legend()
        plt.show()
        plt.savefig('Stock_Price_Prediction.png')


if __name__=='__main__':
    main = MTS_lstm()
    # 数据处理
    df = main.load_data()
    trainX, trainY, testX, testY, scaler = main.process_data(df)
    # 训练
    model = main.run_train(trainX, trainY, testX, testY)
    main.run_val(model,scaler)
