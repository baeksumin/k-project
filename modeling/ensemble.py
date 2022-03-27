
import os

os.system('pip install backtrader')
os.system('pip install backtesting')
os.system('pip install matplotlib')
os.system('pip install tensorflow')
os.system('pip install xgboost')
os.system('pip install s3fs')

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
                size = int(self.broker.getcash() / self.dataclose[0])
                #Keep track of the created order to avoid a 2nd order
                self.order = self.buy(size = size)
                
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


def backtest_data(data_bt):
    
  check_dtype = data_bt.dtype == 'object'

  if (check_dtype):   
    return data_bt.str.replace(',','').astype('float')
  else :
    return data_bt.astype('float') 

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    files = glob('/opt/ml/processing/input/*.csv')
    
    
    print("각 인스턴스 처리하는 파일 개수 : ", len(files))    
    
    
    for k, f in enumerate(files):

        code = f[25:]

        mktcap_data = pd.read_csv(f)        
        print(k+1,'번째',code)
                
        
        #TREND 조건 성립이 안되어서 TREDN 열 안만들어진 경우 
        if 'TREND' not in mktcap_data.columns :   
            print(code+'TREND 열 존재하지 않음.')
            continue
            
            
        mktcap_data['TREND'] = mktcap_data['TREND'].fillna(0)            
    

        #최신 데이터가 뒤로 오도록 정렬
        mktcap_data = mktcap_data.sort_values(by=['TRD_DD'])
        mktcap_data.set_index('TRD_DD',drop=True,inplace=True)
        mktcap_data.reset_index('TRD_DD',drop=False,inplace=True)

        Y_colname = ['TREND'] #종속변수
        X_remove = ['TRD_DD']
        X_colname = [x for x in mktcap_data.columns if x not in Y_colname + X_remove] #입력변수들

        X_train, X_test, Y_train, Y_test = datasplit(mktcap_data, Y_colname, X_colname)
        
        
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

        #final_df_rf = pd.DataFrame([], columns = ['full_code', 'train', 'test', 'min_samples_split', 'random_state', 'n_estimators', 'max_depth', 'MDD', 'CAGR', 'profit'])
        final_df_rf = pd.DataFrame()
        final_df_rf['full_code'] = np.nan
        final_df_rf['train'] = np.nan
        final_df_rf['test'] = np.nan    
        final_df_rf['min_samples_split'] = np.nan
        final_df_rf['random_state'] = np.nan
        final_df_rf['n_estimators'] = np.nan
        final_df_rf['max_depth'] = np.nan
        final_df_rf['MDD'] = np.nan
        final_df_rf['CAGR'] = np.nan    
        final_df_rf['profit'] = np.nan        

        #final_df_xgb = pd.DataFrame([], columns = ['full_code', 'train', 'test', 'reg_alpha', 'random_state', 'n_estimators', 'max_depth', 'MDD', 'CAGR', 'profit'])
        final_df_xgb = pd.DataFrame()
        final_df_xgb['full_code'] = np.nan
        final_df_xgb['train'] = np.nan  
        final_df_xgb['test'] = np.nan 
        final_df_xgb['reg_alpha'] = np.nan     
        final_df_xgb['random_state'] = np.nan     
        final_df_xgb['n_estimators'] = np.nan     
        final_df_xgb['max_depth'] = np.nan     
        final_df_xgb['MDD'] = np.nan     
        final_df_xgb['CAGR'] = np.nan     
        final_df_xgb['profit'] = np.nan       

        #final_df = pd.DataFrame([], columns = ['full_code', 'train', 'test', 'MDD', 'CAGR', 'profit'])
        final_df = pd.DataFrame()    
        final_df['full_code'] = np.nan   
        final_df['train'] = np.nan       
        final_df['test'] = np.nan       
        final_df['MDD'] = np.nan       
        final_df['CAGR'] = np.nan       
        final_df['profit'] = np.nan       
        
        
        #optimum_df = pd.DataFrame([], columns = ['train', 'test', 'max_iter', 'random_state', 'l1_ratio', 'alpha', 'MDD', 'CAGR', 'profit'])
        #optimum_df = pd.DataFrame([], columns = ['train', 'test', 'MDD', 'CAGR', 'profit'])
        optimum_df = pd.DataFrame()
        optimum_df['train']= np.nan
        optimum_df['test']= np.nan
        optimum_df ['max_iter']= np.nan
        optimum_df['random_state']= np.nan
        optimum_df['l1_ratio']= np.nan
        optimum_df['alpha']= np.nan
        optimum_df['MDD']= np.nan
        optimum_df['CAGR']= np.nan
        optimum_df['profit']= np.nan

        
        
        headers = ['train', 'test', 'max_iter', 'random_state', 'l1_ratio', 'alpha', 'MDD', 'CAGR', 'profit']
        #profit_df = pd.DataFrame([], columns = headers)
        
        profit_df = pd.DataFrame()
        profit_df['train']= np.nan
        profit_df['test']= np.nan
        profit_df['max_iter']= np.nan        
        profit_df['random_state']= np.nan         
        profit_df['l1_ratio']= np.nan         
        profit_df['alpha']= np.nan         
        profit_df['MDD']= np.nan         
        profit_df['CAGR']= np.nan         
        profit_df['profit']= np.nan            
        
        
        for mi in [5000000]:
            for rs in [42]:
                for l1 in [0.3,0.4,0.5,0.6,0.7]:
                      for al in [0.00001,0.0001,0.001,0.002,0.004,0.006,0.008,0.01]:
                          model = ElasticNet(
                              max_iter = mi,
                              random_state = rs,
                              l1_ratio = l1,
                              alpha = al,)
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
                          #print('\n', test_score.value_counts(['pred']), '--------------------------------')

                          #df = pd.read_json(json_data[mktcap_top], orient ='index') # 첫번째 키값으로 데이터프레임 변환                 
                          #json_df = df.transpose()
                        
                          file = 's3://sagemaker-hhkim/data/'+code
                          json_df = file.format('ap-northeast-2')
                          json_df = pd.read_csv(json_df)
                        
                          #c_num=code[:21]+code[-21:] 
                          #json_df = c_num.format(region)
                          #json_df = pd.read_csv(json_df)  

                          json_df = json_df[['TRD_DD', 'TDD_OPNPRC', 'TDD_HGPRC', 'TDD_LWPRC', 'TDD_CLSPRC', 'ACC_TRDVOL']]
                          json_df = json_df.sort_values(by=['TRD_DD'])
                          json_df.reset_index(drop=True,inplace=True)

                          df_end = json_df.index[json_df['TRD_DD'] == '2021/12/30'][0]
                          raw = json_df.iloc[a[0]:df_end+1]
                          raw.reset_index(drop = True, inplace = True)
                          raw['TRD_DD'] = pd.to_datetime(raw['TRD_DD'])

                          final = pd.merge(raw, test_score, left_index = True, right_index = True).drop(columns='TRD_DD_y')
                          final = final.rename(columns={'TRD_DD_x': 'TRD_DD'})
                          final['TDD_OPNPRC'] = backtest_data(final['TDD_OPNPRC'])
                          final['TDD_HGPRC'] = backtest_data(final['TDD_HGPRC'])
                          final['TDD_LWPRC'] = backtest_data(final['TDD_LWPRC'])
                          final['TDD_CLSPRC'] = backtest_data(final['TDD_CLSPRC'])
                          final['ACC_TRDVOL'] = backtest_data(final['ACC_TRDVOL'])
                          final = final.astype({'pred':'int'})


                          # 세레브로 (벡트레이더 엔진) 설정

                          # 세레브로 가져오기
                          cerebro = bt.Cerebro()

                          # 데이터 가져오기
                          data = PandasData_Signal(dataname = final,
                                          datetime = 0,
                                          open = 1,
                                          high = 2,
                                          low = 3,
                                          close = 4,
                                          volume = 5,
                                          openinterest = -1,
                                          signal = 6,
                                          )

                          # 데이터 추가하기
                          cerebro.adddata(data)

                          # 전략 추가하기
                          cerebro.addstrategy(MLSignal)

                          # 브로커 설정
                          cerebro.broker.setcash(10000000)

                          # 매매 단위 설정
                          #cerebro.addsizer(bt.sizers.SizerFix, stake = 30) #한번에 30주 설정.

                          # 3. 세레브로 실행하기

                          # 초기 투자금 가져오기
                          init_cash = cerebro.broker.getvalue()

                          # 평가지표 추가
                          cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
                          cerebro.addanalyzer(bt.analyzers.Returns)
                          cerebro.addanalyzer(bt.analyzers.DrawDown)

                          # 세레브로 실행하기
                          results = cerebro.run()

                          # 최종 금액 가져오기
                          final_cash = cerebro.broker.getvalue()

                          analyzers = results[0].analyzers
                          mdd = analyzers.drawdown.get_analysis()['max']['drawdown']
                          cagr = analyzers.returns.get_analysis()['rnorm100']
                          # sharp = analyzers.sharperatio.get_analysis()['sharperatio']
                          print(f"mdd : {mdd:.2f}, cagr: {cagr:.2f}")

                          # print('init_cash: ', init_cash)
                          # print('final_cash: ', final_cash)
                            
                          profit_check = float(final_cash - init_cash)
                        
                          if (profit_check != 0):
                                profit = profit_check / float(init_cash) * 100.
                          else :
                            profit = 0

                          #profit = float(final_cash - init_cash)/float(init_cash) * 100.
                          print("최종금액 : ", final_cash, "원")
                          print("수익률 : ", profit, "%")
                          print("\n")

                          profit_list = [a[0], b[0], mi, rs, l1, al, mdd, cagr, profit]
                          profit_df = profit_df.append(pd.Series(profit_list, index = headers), ignore_index = True)
                          # print('profit_df')
                          # print(profit_df)

        max_profit = profit_df[profit_df['profit'] == profit_df['profit'].max()].reset_index(drop = True)


        num_max_profit = pd.concat([max_profit], axis = 1)
        optimum_df = pd.concat([optimum_df, num_max_profit], axis = 0).reset_index(drop = True)

        #print(optimum_df)
        
        op_list = optimum_df.iloc[0].tolist()
        op_list.insert(0, code[:-4])

        
        #print(op_list)

        final_df_el = final_df_el.append(pd.Series(op_list, index=final_df_el.columns), ignore_index=True)
        print(final_df_el)
        
        
        print("\n\n------------------------------------------------------------------------------------------\n\n\n")
        #optimum_df = pd.DataFrame([], columns = ['train', 'test', 'min_samples_split', 'random_state', 'n_estimators', 'max_depth', 'MDD', 'CAGR', 'profit'])
        optimum_df = pd.DataFrame()
        optimum_df['train']= np.nan
        optimum_df['test']= np.nan
        optimum_df ['min_samples_split']= np.nan
        optimum_df['random_state']= np.nan
        optimum_df['n_estimators']= np.nan
        optimum_df['max_depth']= np.nan
        optimum_df['MDD']= np.nan
        optimum_df['CAGR']= np.nan
        optimum_df['profit']= np.nan
        
        headers = ['train', 'test', 'min_samples_split', 'random_state', 'n_estimators', 'max_depth', 'MDD', 'CAGR', 'profit']
        
        #profit_df = pd.DataFrame([], columns = headers)
        profit_df = pd.DataFrame()
        profit_df['train']= np.nan
        profit_df['test']= np.nan
        profit_df['min_samples_split']= np.nan        
        profit_df['random_state']= np.nan         
        profit_df['n_estimators']= np.nan         
        profit_df['max_depth']= np.nan         
        profit_df['MDD']= np.nan         
        profit_df['CAGR']= np.nan         
        profit_df['profit']= np.nan  
        
        for mi in [2,3,4]:
            for rs in [42]:
                for ne in [110,130,150,170]:
                    for md in [25,30,35,40]:
                          model = RandomForestRegressor(
                              min_samples_split = mi,
                              random_state = rs,
                              n_estimators = ne,
                              max_depth = md,)
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
                          #print('\n', test_score.value_counts(['pred']), '--------------------------------')

                          file = 's3://sagemaker-hhkim/data/'+code
                          json_df = file.format('ap-northeast-2')
                          json_df = pd.read_csv(json_df)

                          json_df = json_df[['TRD_DD', 'TDD_OPNPRC', 'TDD_HGPRC', 'TDD_LWPRC', 'TDD_CLSPRC', 'ACC_TRDVOL']]
                          json_df = json_df.sort_values(by=['TRD_DD'])
                          json_df.reset_index(drop=True,inplace=True)                          

                          df_end = json_df.index[json_df['TRD_DD'] == '2021/12/30'][0]
                          raw = json_df.iloc[a[0]:df_end+1]
                          raw.reset_index(drop = True, inplace = True)
                          raw['TRD_DD'] = pd.to_datetime(raw['TRD_DD'])

                          final = pd.merge(raw, test_score, left_index = True, right_index = True).drop(columns='TRD_DD_y')
                          final = final.rename(columns={'TRD_DD_x': 'TRD_DD'})
                          final['TDD_OPNPRC'] = backtest_data(final['TDD_OPNPRC'])
                          final['TDD_HGPRC'] = backtest_data(final['TDD_HGPRC'])
                          final['TDD_LWPRC'] = backtest_data(final['TDD_LWPRC'])
                          final['TDD_CLSPRC'] = backtest_data(final['TDD_CLSPRC'])
                          final['ACC_TRDVOL'] = backtest_data(final['ACC_TRDVOL'])
                          final = final.astype({'pred':'int'})
                          # print(final.isnull().sum())
                          # print('3--------------------------------------------------')


                          # 세레브로 (벡트레이더 엔진) 설정

                          # 세레브로 가져오기
                          cerebro = bt.Cerebro()

                          # 데이터 가져오기
                          data = PandasData_Signal(dataname = final,
                                          datetime = 0,
                                          open = 1,
                                          high = 2,
                                          low = 3,
                                          close = 4,
                                          volume = 5,
                                          openinterest = -1,
                                          signal = 6,
                                          )

                          # 데이터 추가하기
                          cerebro.adddata(data)

                          # 전략 추가하기
                          cerebro.addstrategy(MLSignal)

                          # 브로커 설정
                          cerebro.broker.setcash(10000000)

                          # 매매 단위 설정
                          #cerebro.addsizer(bt.sizers.SizerFix, stake = 30) #한번에 30주 설정.

                          # 3. 세레브로 실행하기

                          # 초기 투자금 가져오기
                          init_cash = cerebro.broker.getvalue()

                          # 평가지표 추가
                          cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
                          cerebro.addanalyzer(bt.analyzers.Returns)
                          cerebro.addanalyzer(bt.analyzers.DrawDown)

                          # 세레브로 실행하기
                          results = cerebro.run()

                          # 최종 금액 가져오기
                          final_cash = cerebro.broker.getvalue()

                          analyzers = results[0].analyzers
                          mdd = analyzers.drawdown.get_analysis()['max']['drawdown']
                          cagr = analyzers.returns.get_analysis()['rnorm100']
                          # sharp = analyzers.sharperatio.get_analysis()['sharperatio']
                          print(f"mdd : {mdd:.2f}, cagr: {cagr:.2f}")

                          # print('init_cash: ', init_cash)
                          # print('final_cash: ', final_cash)

                          profit_check = float(final_cash - init_cash)
                        
                          if (profit_check != 0):
                                profit = profit_check / float(init_cash) * 100.
                          else :
                            profit = 0                            
                                            
                            
                          #profit = float(final_cash - init_cash)/float(init_cash) * 100.
                          print("최종금액 : ", final_cash, "원")
                          print("수익률 : ", profit, "%")
                          print("\n")

                          profit_list = [a[0], b[0], mi, rs, ne, md, mdd, cagr, profit]
                          profit_df = profit_df.append(pd.Series(profit_list, index = headers), ignore_index = True)
                          # print('profit_df')
                          # print(profit_df)

        max_profit = profit_df[profit_df['profit'] == profit_df['profit'].max()].reset_index(drop = True)


        num_max_profit = pd.concat([max_profit], axis = 1)
        optimum_df = pd.concat([optimum_df, num_max_profit], axis = 0).reset_index(drop = True)
          # print(optimum_df)

        op_list = optimum_df.iloc[0].tolist()
        op_list_rf = op_list.copy()
        op_list.insert(0, code[:-4])
        print(op_list)

        final_df_rf = final_df_rf.append(pd.Series(op_list, index=final_df_rf.columns), ignore_index=True)
        print(final_df_rf)
        
        
        
        print("\n\n------------------------------------------------------------------------------------------\n\n\n")
        #optimum_df = pd.DataFrame([], columns = ['train', 'test', 'reg_alpha', 'random_state', 'n_estimators', 'max_depth', 'MDD', 'CAGR', 'profit'])
        optimum_df = pd.DataFrame()
        optimum_df['train']= np.nan
        optimum_df['test']= np.nan
        optimum_df ['reg_alpha']= np.nan
        optimum_df['random_state']= np.nan
        optimum_df['n_estimators']= np.nan
        optimum_df['max_depth']= np.nan
        optimum_df['MDD']= np.nan
        optimum_df['CAGR']= np.nan
        optimum_df['profit']= np.nan
        
        headers = ['train', 'test', 'reg_alpha', 'random_state', 'n_estimators', 'max_depth', 'MDD', 'CAGR', 'profit']
        
        #profit_df = pd.DataFrame([], columns = headers)
        profit_df = pd.DataFrame()
        profit_df['train']= np.nan
        profit_df['test']= np.nan
        profit_df['reg_alpha']= np.nan        
        profit_df['random_state']= np.nan         
        profit_df['n_estimators']= np.nan         
        profit_df['max_depth']= np.nan         
        profit_df['MDD']= np.nan         
        profit_df['CAGR']= np.nan         
        profit_df['profit']= np.nan  
        
        
        for ra in [1e-2,5e-2,1e-1]:
            for rs in [42]:
                for ne in [110,130,150,170]:
                    for md in [25,30,35,40]:
                          model = XGBRegressor(
                              reg_alpha = ra,
                              random_state = rs,
                              n_estimators = ne,
                              max_depth = md,)
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
                          #print('\n', test_score.value_counts(['pred']), '--------------------------------')

                          file = 's3://sagemaker-hhkim/data/'+code
                          json_df = file.format('ap-northeast-2')
                          json_df = pd.read_csv(json_df)

                          json_df = json_df[['TRD_DD', 'TDD_OPNPRC', 'TDD_HGPRC', 'TDD_LWPRC', 'TDD_CLSPRC', 'ACC_TRDVOL']]
                          json_df = json_df.sort_values(by=['TRD_DD'])
                          json_df.reset_index(drop=True,inplace=True)     

                          df_end = json_df.index[json_df['TRD_DD'] == '2021/12/30'][0]
                          raw = json_df.iloc[a[0]:df_end+1]
                          raw.reset_index(drop = True, inplace = True)
                          raw['TRD_DD'] = pd.to_datetime(raw['TRD_DD'])

                          final = pd.merge(raw, test_score, left_index = True, right_index = True).drop(columns='TRD_DD_y')
                          
                          final = final.rename(columns={'TRD_DD_x': 'TRD_DD'})
                            
                          
                          final['TDD_OPNPRC'] = backtest_data(final['TDD_OPNPRC'])
                          final['TDD_HGPRC'] = backtest_data(final['TDD_HGPRC'])
                          final['TDD_LWPRC'] = backtest_data(final['TDD_LWPRC'])
                          final['TDD_CLSPRC'] = backtest_data(final['TDD_CLSPRC'])
                          final['ACC_TRDVOL'] = backtest_data(final['ACC_TRDVOL'])
        
                          final = final.astype({'pred':'int'})
                          # print(final.isnull().sum())
                          # print('3--------------------------------------------------')


                          # 세레브로 (벡트레이더 엔진) 설정

                          # 세레브로 가져오기
                          cerebro = bt.Cerebro()

                          # 데이터 가져오기
                          data = PandasData_Signal(dataname = final,
                                          datetime = 0,
                                          open = 1,
                                          high = 2,
                                          low = 3,
                                          close = 4,
                                          volume = 5,
                                          openinterest = -1,
                                          signal = 6,
                                          )

                          # 데이터 추가하기
                          cerebro.adddata(data)

                          # 전략 추가하기
                          cerebro.addstrategy(MLSignal)

                          # 브로커 설정
                          cerebro.broker.setcash(10000000)

                          # 매매 단위 설정
                          #cerebro.addsizer(bt.sizers.SizerFix, stake = 30) #한번에 30주 설정.

                          # 3. 세레브로 실행하기

                          # 초기 투자금 가져오기
                          init_cash = cerebro.broker.getvalue()

                          # 평가지표 추가
                          cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
                          cerebro.addanalyzer(bt.analyzers.Returns)
                          cerebro.addanalyzer(bt.analyzers.DrawDown)

                          # 세레브로 실행하기
                          results = cerebro.run()

                          # 최종 금액 가져오기
                          final_cash = cerebro.broker.getvalue()

                          analyzers = results[0].analyzers
                          mdd = analyzers.drawdown.get_analysis()['max']['drawdown']
                          cagr = analyzers.returns.get_analysis()['rnorm100']
                          # sharp = analyzers.sharperatio.get_analysis()['sharperatio']
                          print(f"mdd : {mdd:.2f}, cagr: {cagr:.2f}")

                          # print('init_cash: ', init_cash)
                          # print('final_cash: ', final_cash)

                          profit_check = float(final_cash - init_cash)
                        
                          if (profit_check != 0):
                                profit = profit_check / float(init_cash) * 100.
                          else :
                            profit = 0                            
                                                                                    
                          #profit = float(final_cash - init_cash)/float(init_cash) * 100.
                          print("최종금액 : ", final_cash, "원")
                          print("수익률 : ", profit, "%")
                          print("\n")

                          profit_list = [a[0], b[0], ra, rs, ne, md, mdd, cagr, profit]
                          profit_df = profit_df.append(pd.Series(profit_list, index = headers), ignore_index = True)
                          # print('profit_df')
                          # print(profit_df)

        max_profit = profit_df[profit_df['profit'] == profit_df['profit'].max()].reset_index(drop = True)


        num_max_profit = pd.concat([max_profit], axis = 1)
        optimum_df = pd.concat([optimum_df, num_max_profit], axis = 0).reset_index(drop = True)
          # print(optimum_df)

        op_list = optimum_df.iloc[0].tolist()
        op_list_xgb = op_list.copy()
        op_list.insert(0, code[:-4])
        print(op_list)

        final_df_xgb = final_df_xgb.append(pd.Series(op_list, index=final_df_xgb.columns), ignore_index=True)

        print(final_df_xgb)             

        print("\n\n------------------------------------------------------------------------------------------\n\n\n")
        #optimum_df = pd.DataFrame([], columns = ['train', 'test', 'MDD', 'CAGR', 'profit'])
        optimum_df = pd.DataFrame()
        optimum_df['train']= np.nan
        optimum_df['test']= np.nan
        optimum_df ['MDD']= np.nan
        optimum_df['CAGR']= np.nan
        optimum_df['profit']= np.nan

        
        headers = ['train', 'test', 'MDD', 'CAGR', 'profit']
        
        #profit_df = pd.DataFrame([], columns = headers)
        profit_df = pd.DataFrame()
        profit_df['train']= np.nan
        profit_df['test']= np.nan
        profit_df['MDD']= np.nan        
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
        #print('\n', test_score.value_counts(['pred']), '--------------------------------')

        file = 's3://sagemaker-hhkim/data/'+code
        json_df = file.format('ap-northeast-2')
        json_df = pd.read_csv(json_df)

        json_df = json_df[['TRD_DD', 'TDD_OPNPRC', 'TDD_HGPRC', 'TDD_LWPRC', 'TDD_CLSPRC', 'ACC_TRDVOL']]
        json_df = json_df.sort_values(by=['TRD_DD'])
        json_df.reset_index(drop=True,inplace=True)     

        df_end = json_df.index[json_df['TRD_DD'] == '2021/12/30'][0]
        raw = json_df.iloc[a[0]:df_end+1]
        raw.reset_index(drop = True, inplace = True)
        raw['TRD_DD'] = pd.to_datetime(raw['TRD_DD'])


        final = pd.merge(raw, test_score, left_index = True, right_index = True).drop(columns='TRD_DD_y')
        final = final.rename(columns={'TRD_DD_x': 'TRD_DD'})
        
        final['TDD_OPNPRC'] = backtest_data(final['TDD_OPNPRC'])
        final['TDD_HGPRC'] = backtest_data(final['TDD_HGPRC'])
        final['TDD_LWPRC'] = backtest_data(final['TDD_LWPRC'])
        final['TDD_CLSPRC'] = backtest_data(final['TDD_CLSPRC'])
        final['ACC_TRDVOL'] = backtest_data(final['ACC_TRDVOL'])
        final = final.astype({'pred':'int'})



                          # 세레브로 (벡트레이더 엔진) 설정

                          # 세레브로 가져오기
        cerebro = bt.Cerebro()

                          # 데이터 가져오기
        data = PandasData_Signal(dataname = final,
                                          datetime = 0,
                                          open = 1,
                                          high = 2,
                                          low = 3,
                                          close = 4,
                                          volume = 5,
                                          openinterest = -1,
                                          signal = 6,
                                          )

                          # 데이터 추가하기
        cerebro.adddata(data)

                          # 전략 추가하기
        cerebro.addstrategy(MLSignal)

                          # 브로커 설정
        cerebro.broker.setcash(10000000)

                          # 매매 단위 설정
        #cerebro.addsizer(bt.sizers.SizerFix, stake = 30) #한번에 30주 설정.

                          # 3. 세레브로 실행하기

                          # 초기 투자금 가져오기
        init_cash = cerebro.broker.getvalue()

                          # 평가지표 추가
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.Returns)
        cerebro.addanalyzer(bt.analyzers.DrawDown)

                          # 세레브로 실행하기
        results = cerebro.run()

                          # 최종 금액 가져오기
        final_cash = cerebro.broker.getvalue()

        analyzers = results[0].analyzers
        mdd = analyzers.drawdown.get_analysis()['max']['drawdown']
        cagr = analyzers.returns.get_analysis()['rnorm100']
        print(f"mdd : {mdd:.2f}, cagr: {cagr:.2f}")


        #profit = float(final_cash - init_cash)/float(init_cash) * 100.
        profit_check = float(final_cash - init_cash)

        if (profit_check != 0):
            profit = profit_check / float(init_cash) * 100.
        else :
            profit = 0      
                
        
        
        print("최종금액 : ", final_cash, "원")
        print("수익률 : ", profit, "%")
        print("\n")

        profit_list = [a[0], b[0], mdd, cagr, profit]
        profit_df = profit_df.append(pd.Series(profit_list, index = headers), ignore_index = True)

        max_profit = profit_df[profit_df['profit'] == profit_df['profit'].max()].reset_index(drop = True)


        num_max_profit = pd.concat([max_profit], axis = 1)
        optimum_df = pd.concat([optimum_df, num_max_profit], axis = 0).reset_index(drop = True)
        # print(optimum_df)

        op_list = optimum_df.iloc[0].tolist()
        op_list.insert(0, code[:-4])
        #print(op_list)

        final_df = final_df.append(pd.Series(op_list, index=final_df.columns), ignore_index=True)

        #print(final_df)
        
        
        model = final_df_el.copy()       
        model.iloc[0,0]='ElasticNet'
        
        op_list_rf.insert(0,'randomforest')
        model = model.append(pd.Series(op_list_rf, index=final_df_el.columns),ignore_index=True )
        
        op_list_xgb.insert(0,'xgboost')
        model = model.append(pd.Series(op_list_xgb, index=final_df_el.columns),ignore_index=True )        
  
        model.rename(columns = {'full_code' : 'model'}, inplace = True)         
        
        print("\n\n\n===========================================================================================================\n")
        print('ElasticNet')
        print(final_df_el)
        
        print('\n\nrandomforest')
        print(final_df_rf)
        
        print('\n\nxgboost')
        print(final_df_xgb)        
        
        print('\n\nmodel')
        print(model)            
        
        print('\n\nFinal')       
        print(final_df)
        
        output_path_model = os.path.join('/opt/ml/processing/model' , code)
        pd.DataFrame(model).to_csv(output_path_model, index=False)
        print('Saving train data {}'.format(output_path_model))
        
        output_path_ensemble = os.path.join('/opt/ml/processing/ensemble' , code)
        pd.DataFrame(final_df).to_csv(output_path_ensemble, index=False)
        print('Saving train data {}'.format(output_path_ensemble))        
        
    #print("--------------------------------------")      
    #f = 's3://sagemaker-hhkim/train/ElasticNet.csv'
    #df = f.format('ap-northeast-2')
    #df = pd.read_csv(df)
    
    #print(df)
    #append_df = df.append(final_df_el)
    
    #print(append_df)
    #output_path = os.path.join('/opt/ml/processing/train_data' , 'ElasticNet.csv')

    #pd.DataFrame(append_df).to_csv(output_path, index=False)
    #print('Saving train data {}'.format(output_path))
       



