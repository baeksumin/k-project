
import os

os.system('pip install backtrader')
os.system('pip install backtesting')
os.system('pip install matplotlib')
os.system('pip install tensorflow')
os.system('pip install xgboost')

os.system('pip install --upgrade scikit-learn')

import argparse
import warnings
from glob import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import LSTM, SimpleRNN, GRU, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import backtrader as bt
from backtrader.feeds import PandasData
import backtrader.indicators as btind

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class PandasData_Signal(PandasData):

  lines = ('signal',)

  # add the parameter to the parameters inherited from the base class

  params = (('signal', 6),)

class MLSignal(bt.SignalStrategy):

    def log(self, txt, dt=None):

        ''' Logging function fot this strategy'''

        dt = dt or self.datas[0].datetime.date(0)

        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.signal = self.datas[0].signal
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                    order.executed.value,
                    order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                        (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                (trade.pnl, trade.pnlcomm))
        
    def next(self):
        #Check if we are in the market 
        if not self.position :            
            if self.signal[0] == -1:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                #Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
                
        else:           
            #Already in the market...we might sell                     
            if self.signal[0] == 1:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()
                
                
def datasplit(df, Y_colname, X_colname): #데이터 분리 함수
    
  df_train = df.loc[0 : round(len(df) * 0.75)] 
  df_test = df.loc[round(len(df) * 0.75) + 1 : ]
  Y_train = df_train[Y_colname]
  X_train = df_train[X_colname]
  Y_test = df_test[Y_colname]
  X_test = df_test[X_colname]
    
  return X_train, X_test, Y_train, Y_test


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    files = glob('/opt/ml/processing/input/*.csv')
    
    
    print("각 인스턴스 처리하는 파일 개수 : ", len(files))    
       
    
    #final_df = pd.DataFrame([], columns = ['full_code', 'train', 'test', 'MDD', 'CAGR', 'profit'])
    final_df = pd.DataFrame()
    final_df['full_code'] = np.nan
    final_df['train'] = np.nan
    final_df['test'] = np.nan
    final_df['MDD'] = np.nan
    final_df['CAGR'] = np.nan
    final_df['profit'] = np.nan

    
    for k, f in enumerate(files):

        code = f[25:]

        mktcap_data = pd.read_csv(f)        
        print(k+1,'번째',code)

        mktcap_data['TREND'] = mktcap_data['TREND'].fillna(0)        
        
        
        #최신 데이터가 뒤로 오도록 정렬
        mktcap_data = mktcap_data.sort_values(by=['TRD_DD'])
        mktcap_data.set_index('TRD_DD',drop=True,inplace=True)
        mktcap_data.reset_index('TRD_DD',drop=False,inplace=True)        
                
        
        #final_df_el = pd.DataFrame([], columns = ['full_code', 'train', 'test', 'max_iter', 'random_state', 'l1_ratio', 'alpha', 'MDD', 'CAGR', 'profit'])
        final_df_el = pd.DataFrame()
        final_df_el['full_code'] = np.nan
        final_df_el['train'] = np.nan
        final_df_el['test'] = np.nan
        final_df_el['max_iter'] = np.nan
        final_df_el['random_state'] = np.nan
        final_df_el['l1_ratio'] = np.nan
        final_df_el['alpha'] = np.nan
        final_df_el['MDD'] = np.nan
        final_df_el['CAGR'] = np.nan
        final_df_el['profit'] = np.nan    
    
            
        
        Y_colname = ['TREND'] #종속변수
        X_remove = ['TRD_DD']
        X_colname = [x for x in mktcap_data.columns if x not in Y_colname + X_remove] #입력변수들

        X_train, X_test, Y_train, Y_test = datasplit(mktcap_data, Y_colname, X_colname)

        #optimum_df = pd.DataFrame([], columns = ['train', 'test', 'MDD', 'CAGR', 'profit'])
        optimum_df = pd.DataFrame()
        optimum_df['train']= np.nan
        optimum_df['test']= np.nan
        optimum_df ['MDD']= np.nan
        optimum_df['CAGR']= np.nan
        optimum_df['profit']= np.nan

        #headers = ['train', 'test', 'MDD', 'CAGR', 'profit']
        #profit_df = pd.DataFrame([], columns = headers)
        profit_df = pd.DataFrame()
        profit_df['train']= np.nan
        profit_df['test']= np.nan
        profit_df ['MDD']= np.nan
        profit_df['CAGR']= np.nan
        profit_df['profit']= np.nan
        
        
        en_model = ElasticNet(max_iter=int(final_df_el['max_iter'][0]), random_state=int(final_df_el['random_state'][0]), l1_ratio=final_df_el['l1_ratio'][0], alpha=final_df_el['alpha'][0])
        rf_model = RandomForestRegressor(min_samples_split=int(final_df_rf['min_samples_split'][0]), random_state=int(final_df_rf['random_state'][0]), n_estimators=int(final_df_rf['n_estimators'][0]), max_depth=int(final_df_rf['max_depth'][0]))
        xgb_model = XGBRegressor(reg_alpha=final_df_xgb['reg_alpha'][0], random_state=int(final_df_xgb['random_state'][0]), n_estimators=int(final_df_xgb['n_estimators'][0]), max_depth=int(final_df_xgb['max_depth'][0]))

        model = VotingRegressor(estimators=[('xgb', xgb_model), ('rf', rf_model), ('en', en_model)])
        model.fit(X_train, Y_train)

        past = model.predict(X_test)

        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)


        Y_test_array = np.array(Y_test).reshape(-1,1)
        real = list(Y_test_array.flatten())
        Y_test_pred_array = np.array(Y_test_pred).reshape(-1,1)
        pred = list(Y_test_pred_array.flatten())
        
        a = list(X_train.shape)
        b = list(X_test.shape)
        test_data = mktcap_data.iloc[a[0]:]

        test_score = test_data.copy()
        test_score['pred'] = Y_test_pred
        test_score = test_score[['TRD_DD', 'pred']]
        test_score = test_score.reset_index()
        test_score = test_score.drop(['index'], axis = 1)

        for j in range(len(test_score)):
          if test_score['pred'][j] < -0.1:
            test_score['pred'][j] = -1
          elif test_score['pred'][j] > 0.1:
            test_score['pred'][j] = 1
          else:
            test_score['pred'][j] = 0
        print('\n', test_score.value_counts(['pred']), '--------------------------------')


        #output_path = os.path.join('/opt/ml/processing/processed_data' , code)

        #pd.DataFrame(df).to_csv(output_path, index=False)
        #print('Saving train data {}'.format(output_path))
       



