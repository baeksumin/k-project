
import os

os.system('pip install talib-binary')
os.system('pip install seaborn')
os.system('pip install backtrader')
os.system('pip install backtesting')
os.system('pip install deap')
os.system('pip install IPython')
#os.system('pip install natsort')

import argparse
import warnings
import pandas as pd
import numpy as np
from glob import glob
from sklearn.exceptions import DataConversionWarning
import talib
import re
import sys
import json
import time
import pickle
import random
import logging
import seaborn as sns
from tqdm import trange
import backtrader as bt
import matplotlib.pyplot as plt
from backtesting import Strategy
from backtesting import Backtest
import backtrader.feeds as btfeeds
from IPython.display import display, Image
from datetime import datetime, date, timedelta
from deap import base, creator, tools, algorithms
#import natsort

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class SmaCross1(bt.Strategy):
  params = dict(
    pfast=50, # period for the fast moving average
    pslow=200 # period for the slow moving average 
    ) 
  
  def __init__(self):
    sma1 = bt.ind.SMA(period = self.p.pfast) # fast moving average 
    sma2 = bt.ind.SMA(period = self.p.pslow) # slow moving average 
    self.crossover = bt.ind.CrossOver(sma1, sma2) # crossover signal 

  def next(self): 
    if not self.position: # not in the market 
      if self.crossover > 0: # if fast crosses slow to the upside 
        close = self.data.close[0] # 종가 값 
          
        size = int(self.broker.getcash() / close) # 최대 구매 가능 개수 
        self.buy(size=size) # 매수 size = 구매 개수 설정 
    elif self.crossover < 0: # in the market & cross to the downside 
        self.close() # 매도

class RSI(bt.Strategy):
  params = dict(period=26)

  def __init__(self):
    self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)

  def next(self):    
    if not self.position:  #아직 주식을 사지 않았다면

      if self.rsi <30 :
        self.order = self.buy()

    elif self. rsi >70 :
      self.order = self.sell()

class ROC(bt.Strategy):
  params = dict(period=14)

  def __init__(self):
    self.roc = bt.indicators.ROC(self.data.close, period=self.p.period)

  def next(self):    
    if not self.position:  #아직 주식을 사지 않았다면

      if self.roc > 0:
        self.order = self.buy()

    elif self.roc < 0:
      self.order = self.sell()

class MAP(bt.Strategy):
  params = dict(period = 12, upperLimit = .07, lowerLimit = .07)

  def __init__(self):
    # self.sma = bt.ind.SMA(self.data.close, period = self.p.period) # fast moving average 
    self.sma = bt.ind.SMA(period = self.p.period)
    self.ul = self.sma + (self.p.upperLimit * self.sma)
    self.ll = self.sma + (self.p.lowerLimit * self.sma)

  def next(self): 
    if not self.position: # 아직 주식을 사지 않았다면
      if self.sma <= self.ll:
        close = self.data.close[0] # 종가 값             
        size = int(self.broker.getcash() / close) # 최대 구매 가능 개수 
        self.buy(size=size) # 매수 size = 구매 개수 설정 

    elif self.sma > self.ul:
        self.sell() # 매도lf.sell() # 매도

class StochasticSR(bt.Strategy):
    '''Trading strategy that utilizes the Stochastic Oscillator indicator for oversold/overbought entry points, 
    and previous support/resistance via Donchian Channels as well as a max loss in pips for risk levels.'''
    # parameters for Stochastic Oscillator and max loss in pips
    # Donchian Channels to determine previous support/resistance levels will use the given period as well
    # http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Stochastic.php5 for Stochastic Oscillator formula and description
    params = (('period', 14), ('pfast', 3), ('pslow', 3), ('upperLimit', 80), ('lowerLimit', 20), ('stop_pips', .002))

    def __init__(self):
        '''Initializes logger and variables required for the strategy implementation.'''
        # initialize logger for log function (set to critical to prevent any unwanted autologs, not using log objects because only care about logging one thing)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format='%(message)s', level=logging.CRITICAL, handlers=[
            logging.FileHandler("LOG.log"),
            logging.StreamHandler()
            ])

        self.order = None
        self.donchian_stop_price = None
        self.price = None
        self.stop_price = None
        self.stop_donchian = None

        self.stochastic = bt.indicators.Stochastic(self.data, period=self.params.period, period_dfast=self.params.pfast, period_dslow=self.params.pslow, 
        upperband=self.params.upperLimit, lowerband=self.params.lowerLimit)


    def next(self):
        '''Checks to see if Stochastic Oscillator, position, and order conditions meet the entry or exit conditions for the execution of buy and sell orders.'''
        if self.order:
            # if there is a pending order, don't do anything
            return
        if self.position.size == 0:
            # When stochastic crosses back below 80, enter short position.
            if self.stochastic.lines.percD[-1] >= 80 and self.stochastic.lines.percD[0] <= 80:
                # stop price at last support level in self.params.period periods
                self.donchian_stop_price = max(self.data.high.get(size=self.params.period))
                self.order = self.sell()
                # stop loss order for max loss of self.params.stop_pips pips
                self.stop_price = self.buy(exectype=bt.Order.Stop, price=self.data.close[0]+self.params.stop_pips, oco=self.stop_donchian)
                # stop loss order for donchian SR price level
                self.stop_donchian = self.buy(exectype=bt.Order.Stop, price=self.donchian_stop_price, oco=self.stop_price)
            # when stochastic crosses back above 20, enter long position.
            elif self.stochastic.lines.percD[-1] <= 20 and self.stochastic.lines.percD[0] >= 20:
                # stop price at last resistance level in self.params.period periods
                self.donchian_stop_price = min(self.data.low.get(size=self.params.period))
                self.order = self.buy()
                # stop loss order for max loss of self.params.stop_pips pips
                self.stop_price = self.sell(exectype=bt.Order.Stop, price=self.data.close[0]-self.params.stop_pips, oco=self.stop_donchian)
                # stop loss order for donchian SR price level
                self.stop_donchian = self.sell(exectype=bt.Order.Stop, price=self.donchian_stop_price, oco=self.stop_price) 
  
        if self.position.size > 0:
            # When stochastic is above 70, close out of long position
            if (self.stochastic.lines.percD[0] >= 70):
                self.close(oco=self.stop_price)
        if self.position.size < 0:
            # When stochastic is below 30, close out of short position
            if (self.stochastic.lines.percD[0] <= 30):
                self.close(oco=self.stop_price)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    files = glob('/opt/ml/processing/input/*.csv')    
        
    
    print("각 인스턴스 처리하는 파일 개수 : ", len(files))
    
    
    
    for k, f in enumerate(files): 

        code = f[25:]
        df = pd.read_csv(f)
        print(k+1,'번째',code)
        
        #df = pd.read_csv(f)
        #df = df[['TRD_DD','MKTCAP']] #날짜, 시가총액 열만 추출 
        
        #시간순 재정렬
        df = df.sort_values(by=['TRD_DD'])
        df.reset_index(drop=True,inplace=True)
        df['TRD_DD'] = pd.to_datetime(df['TRD_DD']) #datetime변환

        #인풋 데이터 모양 맞춰주기(backtest에 들어갈 데이터 모양)
        df_bt = df[['TRD_DD','TDD_OPNPRC','TDD_HGPRC','TDD_LWPRC','TDD_CLSPRC', 'ACC_TRDVOL']]
        df_bt['TRD_DD'] = pd.to_datetime(df_bt['TRD_DD'])
        df_bt.rename(columns={'TRD_DD':'Date', 'TDD_OPNPRC':'Open', 'TDD_HGPRC':'High','TDD_LWPRC':'Low','TDD_CLSPRC':'Close', 'ACC_TRDVOL':'Volume'}, inplace=True)
        df_bt.set_index('Date',drop=True,inplace=True)

        
        def backtest_data(data_bt):
          check_dtype = data_bt.dtype == 'object'
          if (check_dtype):   
            return data_bt.str.replace(',','').astype('float')
          else :
            return data_bt.astype('float')        
                

        #데이터프레임 콤마(,) 제거 그리고 타입 소수로 변환
        df_bt['Open'] = backtest_data(df_bt['Open'])
        df_bt['High'] = backtest_data(df_bt['High'])
        df_bt['Low'] = backtest_data(df_bt['Low'])
        df_bt['Close'] = backtest_data(df_bt['Close'])
        df_bt['Volume'] = backtest_data(df_bt['Volume'])
        
        
# GDC --------------------------------------------------------------------------

        try:

            random.seed(3)

            PARAM_NAMES = ["pfast", "pslow"]

            NGEN = 5  # 알고리즘 5번 반복.
            NPOP = 100 #인구 초기
            CXPB = 0.5  #교차 전략 
            MUTPB = 0.3  #돌연변이 전략.


            #최소fintness 설정 (fitness값이 작을수록 좋도록 설정)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMin)

            # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            # creator.create("Individual", list, fitness=creator.FitnessMax)

            def evaluate(individual, plot=False, log=False):

              strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

              cerebro = bt.Cerebro(stdstats=False)

              data = bt.feeds.PandasData(dataname = df_bt)

              cerebro.adddata(data)

              initial_capital = 1000000
              cerebro.broker.setcash(initial_capital)

              cerebro.addstrategy(SmaCross1, **strategy_params)

              cerebro.addanalyzer(bt.analyzers.DrawDown)

              cerebro.broker.setcommission(commission=0.0025, margin=False)  #수수료 설정

              strats = cerebro.run()

              profit = cerebro.broker.getvalue() - initial_capital

              if profit == 0:
                return [np.inf]

              # max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"] # max.moneydown - max drawdown value in monetary units
              # fitness = profit / (max_dd if max_dd > 0 else 1)
              fitness = round(1 / profit, 15)

              if log:
                print(f"Starting Portfolio Value: {initial_capital:,.2f}")
                print(f"Final Portfolio Value:  {cerebro.broker.getvalue():,.2f}")
                print(f"Total Profit:       {profit:,.2f}")
                print(f"Profit / Max DD:     {fitness}")

              # if plot:
                # cerebro.plot()

              return [fitness]

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(NPOP), NPOP)

            # crossover strategy
            toolbox.register("mate", tools.cxUniform, indpb=CXPB)
            # mutation strategy
            toolbox.register("mutate", tools.mutUniformInt, low=1, up=151, indpb=0.2)
            # selection strategy
            toolbox.register("select", tools.selTournament, tournsize=3)
            # fitness function
            toolbox.register("evaluate", evaluate)


            # definition of an individual & a population
            toolbox.register("attr_sma1", random.randint, 1, 100)
            toolbox.register("attr_sma2", random.randint, 151, 251) 
            toolbox.register(
              "individual",
              tools.initCycle,
              creator.Individual,
              (
                toolbox.attr_sma1,
                toolbox.attr_sma2,

              ),
            )

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            mean = np.ndarray(NGEN)
            best = np.ndarray(NGEN)
            hall_of_fame = tools.HallOfFame(maxsize=3)

            t = time.perf_counter()
            pop = toolbox.population(n=NPOP)
            for g in trange(NGEN):
              # Select the next generation individuals
              offspring = toolbox.select(pop, len(pop))
              # Clone the selected individuals
              offspring = list(map(toolbox.clone, offspring))

              # Apply crossover on the offspring
              for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                  toolbox.mate(child1, child2)
                  del child1.fitness.values
                  del child2.fitness.values

              # Apply mutation on the offspring
              for mutant in offspring:
                if random.random() < MUTPB:
                  toolbox.mutate(mutant)
                  del mutant.fitness.values

              # Evaluate the individuals with an invalid fitness
              invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
              fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
              for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

              # The population is entirely replaced by the offspring
              pop[:] = offspring
              hall_of_fame.update(pop)
              print(
                "\n HALL OF FAME:\n"
                + "\n".join(
                  [
                    f"  {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                    for _, ind in enumerate(hall_of_fame)
                  ]
                )
              )

              fitnesses = [
                ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
              ]
              mean[g] = np.mean(fitnesses)
              best[g] = np.max(fitnesses)

            end_t = time.perf_counter()
            print(f"Time Elapsed: {end_t - t:,.2f}")

            # 최적의 파라미터 값 출력
            OPTIMISED_STRATEGY_PARAMS = {
              k: v for k, v in zip(PARAM_NAMES, hall_of_fame[0])}
            GDC_params = list(OPTIMISED_STRATEGY_PARAMS.values())
            print('**GDC 파라미터 값: ', GDC_params)
            print('\n')                       
            
        except:
            GDC_params = [50, 200]
        
        
# RSI --------------------------------------------------------------------------

        try:
            random.seed(3)

            PARAM_NAMES = ["period"]

            NGEN = 5  # 알고리즘 5번 반복.
            NPOP = 100 #인구 초기
            CXPB = 0.5  #교차 전략 
            MUTPB = 0.3  #돌연변이 전략.


            #최소fintness 설정 (fitness값이 작을수록 좋도록 설정)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMin)

            # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            # creator.create("Individual", list, fitness=creator.FitnessMax)

            def evaluate(individual, plot=False, log=False):

              strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

              cerebro = bt.Cerebro(stdstats=False)

              data = bt.feeds.PandasData(dataname = df_bt)

              cerebro.adddata(data)

              initial_capital = 1000000
              cerebro.broker.setcash(initial_capital)

              cerebro.addstrategy(RSI, **strategy_params)

              cerebro.addanalyzer(bt.analyzers.DrawDown)

              cerebro.broker.setcommission(commission=0.0025, margin=False)  #수수료 설정

              strats = cerebro.run()

              profit = cerebro.broker.getvalue() - initial_capital

              if profit == 0:
                return [np.inf]

              # max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"] # max.moneydown - max drawdown value in monetary units
              # fitness = profit / (max_dd if max_dd > 0 else 1)
              fitness = round(1 / profit, 15)

              if log:
                print(f"Starting Portfolio Value: {initial_capital:,.2f}")
                print(f"Final Portfolio Value:  {cerebro.broker.getvalue():,.2f}")
                print(f"Total Profit:       {profit:,.2f}")
                print(f"Profit / Max DD:     {fitness}")

              # if plot:
                # cerebro.plot()

              return [fitness]

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(NPOP), NPOP)

            # crossover strategy
            toolbox.register("mate", tools.cxUniform, indpb=CXPB)
            # mutation strategy
            toolbox.register("mutate", tools.mutUniformInt, low=1, up=151, indpb=0.2)
            # selection strategy
            toolbox.register("select", tools.selTournament, tournsize=3)
            # fitness function
            toolbox.register("evaluate", evaluate)


            # definition of an individual & a population
            toolbox.register("attr_period", random.randint, 1, 100)
            toolbox.register(
              "individual",
              tools.initCycle,
              creator.Individual,
              (
                toolbox.attr_period,

              ),
            )

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            mean = np.ndarray(NGEN)
            best = np.ndarray(NGEN)
            hall_of_fame = tools.HallOfFame(maxsize=3)

            t = time.perf_counter()
            pop = toolbox.population(n=NPOP)
            for g in trange(NGEN):
              # Select the next generation individuals
              offspring = toolbox.select(pop, len(pop))
              # Clone the selected individuals
              offspring = list(map(toolbox.clone, offspring))

              # Apply crossover on the offspring
              for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                  toolbox.mate(child1, child2)
                  del child1.fitness.values
                  del child2.fitness.values

              # Apply mutation on the offspring
              for mutant in offspring:
                if random.random() < MUTPB:
                  toolbox.mutate(mutant)
                  del mutant.fitness.values

              # Evaluate the individuals with an invalid fitness
              invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
              fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
              for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

              # The population is entirely replaced by the offspring
              pop[:] = offspring
              hall_of_fame.update(pop)
              print(
                "\n HALL OF FAME:\n"
                + "\n".join(
                  [
                    f"  {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                    for _, ind in enumerate(hall_of_fame)
                  ]
                )
              )

              fitnesses = [
                ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
              ]
              mean[g] = np.mean(fitnesses)
              best[g] = np.max(fitnesses)

            end_t = time.perf_counter()
            print(f"Time Elapsed: {end_t - t:,.2f}")

            # 최적의 파라미터 값 출력
            OPTIMISED_STRATEGY_PARAMS = {
              k: v for k, v in zip(PARAM_NAMES, hall_of_fame[0])}
            RSI_params = list(OPTIMISED_STRATEGY_PARAMS.values())
            print('RSI 파라미터 값: ', RSI_params)
            print('\n')


        except:
            RSI_params = [26]

# ROC --------------------------------------------------------------------------

        try:
        
            random.seed(3)

            PARAM_NAMES = ["period"]

            NGEN = 5  # 알고리즘 5번 반복.
            NPOP = 100 #인구 초기
            CXPB = 0.5  #교차 전략 
            MUTPB = 0.3  #돌연변이 전략.


            #최소fintness 설정 (fitness값이 작을수록 좋도록 설정)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMin)

            # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            # creator.create("Individual", list, fitness=creator.FitnessMax)

            def evaluate(individual, plot=False, log=False):

              strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

              cerebro = bt.Cerebro(stdstats=False)

              data = bt.feeds.PandasData(dataname = df_bt)

              cerebro.adddata(data)

              initial_capital = 1000000
              cerebro.broker.setcash(initial_capital)

              cerebro.addstrategy(ROC, **strategy_params)

              cerebro.addanalyzer(bt.analyzers.DrawDown)

              cerebro.broker.setcommission(commission=0.0025, margin=False)  #수수료 설정

              strats = cerebro.run()

              profit = cerebro.broker.getvalue() - initial_capital

              if profit == 0:
                return [np.inf]

              # max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"] # max.moneydown - max drawdown value in monetary units
              # fitness = profit / (max_dd if max_dd > 0 else 1)
              fitness = round(1 / profit, 15)

              if log:
                print(f"Starting Portfolio Value: {initial_capital:,.2f}")
                print(f"Final Portfolio Value:  {cerebro.broker.getvalue():,.2f}")
                print(f"Total Profit:       {profit:,.2f}")
                print(f"Profit / Max DD:     {fitness}")

              # if plot:
                # cerebro.plot()

              return [fitness]

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(NPOP), NPOP)

            # crossover strategy
            toolbox.register("mate", tools.cxUniform, indpb=CXPB)
            # mutation strategy
            toolbox.register("mutate", tools.mutUniformInt, low=1, up=151, indpb=0.2)
            # selection strategy
            toolbox.register("select", tools.selTournament, tournsize=3)
            # fitness function
            toolbox.register("evaluate", evaluate)


            # definition of an individual & a population
            toolbox.register("attr_period", random.randint, 1, 100)
            toolbox.register(
              "individual",
              tools.initCycle,
              creator.Individual,
              (
                toolbox.attr_period,

              ),
            )

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            mean = np.ndarray(NGEN)
            best = np.ndarray(NGEN)
            hall_of_fame = tools.HallOfFame(maxsize=3)

            t = time.perf_counter()
            pop = toolbox.population(n=NPOP)
            for g in trange(NGEN):
              # Select the next generation individuals
              offspring = toolbox.select(pop, len(pop))
              # Clone the selected individuals
              offspring = list(map(toolbox.clone, offspring))

              # Apply crossover on the offspring
              for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                  toolbox.mate(child1, child2)
                  del child1.fitness.values
                  del child2.fitness.values

              # Apply mutation on the offspring
              for mutant in offspring:
                if random.random() < MUTPB:
                  toolbox.mutate(mutant)
                  del mutant.fitness.values

              # Evaluate the individuals with an invalid fitness
              invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
              fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
              for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

              # The population is entirely replaced by the offspring
              pop[:] = offspring
              hall_of_fame.update(pop)
              print(
                "\n HALL OF FAME:\n"
                + "\n".join(
                  [
                    f"  {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                    for _, ind in enumerate(hall_of_fame)
                  ]
                )
              )

              fitnesses = [
                ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
              ]
              mean[g] = np.mean(fitnesses)
              best[g] = np.max(fitnesses)

            end_t = time.perf_counter()
            print(f"Time Elapsed: {end_t - t:,.2f}")

            # 최적의 파라미터 값 출력
            OPTIMISED_STRATEGY_PARAMS = {
              k: v for k, v in zip(PARAM_NAMES, hall_of_fame[0])}
            ROC_params = list(OPTIMISED_STRATEGY_PARAMS.values())
            print('**ROC 파라미터 값: ', ROC_params)


        except:
            ROC_params = [14]

# MAP --------------------------------------------------------------------------

        try:
            random.seed(3)

            PARAM_NAMES = ["period", "upperLimit", "lowerLimit"]

            NGEN = 5  # 알고리즘 5번 반복.
            NPOP = 100 #인구 초기
            CXPB = 0.5  #교차 전략 
            MUTPB = 0.3  #돌연변이 전략.


            #최소fintness 설정 (fitness값이 작을수록 좋도록 설정)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMin)

            # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            # creator.create("Individual", list, fitness=creator.FitnessMax)

            def evaluate(individual, plot=False, log=False):

              strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

              cerebro = bt.Cerebro(stdstats=False)

              data = bt.feeds.PandasData(dataname = df_bt)

              cerebro.adddata(data)

              initial_capital = 1000000
              cerebro.broker.setcash(initial_capital)

              cerebro.addstrategy(MAP, **strategy_params)

              cerebro.addanalyzer(bt.analyzers.DrawDown)

              cerebro.broker.setcommission(commission=0.0025, margin=False)  #수수료 설정

              strats = cerebro.run()

              #profit = cerebro.broker.getvalue() - initial_capital
              profit = cerebro.broker.getvalue()

              if profit == 0:
                return [np.inf]
              # max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"] # max.moneydown - max drawdown value in monetary units
              # fitness = profit / (max_dd if max_dd > 0 else 1)
              fitness = round(1 / profit, 15)

              if log:
                print(f"Starting Portfolio Value: {initial_capital:,.2f}")
                print(f"Final Portfolio Value:  {cerebro.broker.getvalue():,.2f}")
                print(f"Total Profit:       {profit:,.2f}")
                print(f"Profit / Max DD:     {fitness}")

              # if plot:
                # cerebro.plot()

              return [fitness]

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(NPOP), NPOP)

            # crossover strategy
            toolbox.register("mate", tools.cxUniform, indpb=CXPB)
            # mutation strategy
            toolbox.register("mutate", tools.mutUniformInt, low=1, up=151, indpb=0.2)
            # selection strategy
            toolbox.register("select", tools.selTournament, tournsize=3)
            # fitness function
            toolbox.register("evaluate", evaluate)


            # definition of an individual & a population
            toolbox.register("attr_period", random.randint, 1, 100)
            toolbox.register("attr_upperLimit", random.uniform, 0.05, 0.09)
            toolbox.register("attr_lowerLimit", random.uniform, 0.05, 0.09)

            toolbox.register(
              "individual",
              tools.initCycle,
              creator.Individual,
              (
                toolbox.attr_period,
                toolbox.attr_upperLimit,
                toolbox.attr_lowerLimit,

              ),
            )

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            mean = np.ndarray(NGEN)
            best = np.ndarray(NGEN)
            hall_of_fame = tools.HallOfFame(maxsize=3)

            t = time.perf_counter()
            pop = toolbox.population(n=NPOP)
            for g in trange(NGEN):
              # Select the next generation individuals
              offspring = toolbox.select(pop, len(pop))
              # Clone the selected individuals
              offspring = list(map(toolbox.clone, offspring))

              # Apply crossover on the offspring
              for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                  toolbox.mate(child1, child2)
                  del child1.fitness.values
                  del child2.fitness.values

              # Apply mutation on the offspring
              for mutant in offspring:
                if random.random() < MUTPB:
                  toolbox.mutate(mutant)
                  del mutant.fitness.values

              # Evaluate the individuals with an invalid fitness
              invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
              fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
              for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

              # The population is entirely replaced by the offspring
              pop[:] = offspring
              hall_of_fame.update(pop)
              print(
                "\n HALL OF FAME:\n"
                + "\n".join(
                  [
                    f"  {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                    for _, ind in enumerate(hall_of_fame)
                  ]
                )
              )

              fitnesses = [
                ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
              ]
              mean[g] = np.mean(fitnesses)
              best[g] = np.max(fitnesses)

            end_t = time.perf_counter()
            print(f"Time Elapsed: {end_t - t:,.2f}")

            # 최적의 파라미터 값 출력
            OPTIMISED_STRATEGY_PARAMS = {
              k: v for k, v in zip(PARAM_NAMES, hall_of_fame[0])}
            MAP_params = list(OPTIMISED_STRATEGY_PARAMS.values())
            print('**MAP 파라미터 값: ', MAP_params)


        except:
            MAP_params = [12, 0.07, 0.07]

# STC --------------------------------------------------------------------------
        try:
        
            random.seed(3)

            PARAM_NAMES = ["period","pfast","pslow","upperLimit","lowerLimit"]

            NGEN = 5  # 알고리즘 5번 반복.
            NPOP = 100 #인구 초기
            CXPB = 0.5  #교차 전략 
            MUTPB = 0.3  #돌연변이 전략.


            #최소fintness 설정 (fitness값이 작을수록 좋도록 설정)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMin)

            # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            # creator.create("Individual", list, fitness=creator.FitnessMax)

            def evaluate(individual, plot=False, log=False):

              strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

              cerebro = bt.Cerebro(stdstats=False)

              data = bt.feeds.PandasData(dataname = df_bt, name = i)

              cerebro.adddata(data)

              initial_capital = 1000000
              cerebro.broker.setcash(initial_capital)

              cerebro.addstrategy(StochasticSR, **strategy_params)

              cerebro.addanalyzer(bt.analyzers.DrawDown)

              cerebro.broker.setcommission(commission=0.0025, margin=False)  #수수료 설정

              strats = cerebro.run()

              profit = cerebro.broker.getvalue() - initial_capital

              # max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"] # max.moneydown - max drawdown value in monetary units
              # fitness = profit / (max_dd if max_dd > 0 else 1)
              fitness = round(1 / profit, 15)

              if log:
                print(f"Starting Portfolio Value: {initial_capital:,.2f}")
                print(f"Final Portfolio Value:  {cerebro.broker.getvalue():,.2f}")
                print(f"Total Profit:       {profit:,.2f}")
                print(f"Profit / Max DD:     {fitness}")

              # if plot:
                # cerebro.plot()

              return [fitness]

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(NPOP), NPOP)

            # crossover strategy
            toolbox.register("mate", tools.cxUniform, indpb=CXPB)
            # mutation strategy
            toolbox.register("mutate", tools.mutUniformInt, low=1, up=151, indpb=0.2)
            # selection strategy
            toolbox.register("select", tools.selTournament, tournsize=3)
            # fitness function
            toolbox.register("evaluate", evaluate)


            # definition of an individual & a population
            # 파라미터 개수 및 범위 설정 - toolbox.register
            toolbox.register('attr_period', random.randint, 5, 31) 
            toolbox.register('attr_pfast', random.randint, 2, 21)
            toolbox.register('attr_pslow', random.randint, 2, 21)
            toolbox.register('attr_upperLimit', random.randint, 70, 91)
            toolbox.register('attr_lowerLimit', random.randint, 10, 31)


            toolbox.register(
              "individual",
              tools.initCycle,
              creator.Individual,
              (   # 파라미터 개수 설정
                  toolbox.attr_period,
                  toolbox.attr_pfast,
                  toolbox.attr_pslow,
                  toolbox.attr_upperLimit,
                  toolbox.attr_lowerLimit,
              ),
            )

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            mean = np.ndarray(NGEN)
            best = np.ndarray(NGEN)
            hall_of_fame = tools.HallOfFame(maxsize=3)

            t = time.perf_counter()
            pop = toolbox.population(n=NPOP)
            for g in trange(NGEN):
              # Select the next generation individuals
              offspring = toolbox.select(pop, len(pop))
              # Clone the selected individuals
              offspring = list(map(toolbox.clone, offspring))

              # Apply crossover on the offspring
              for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                  toolbox.mate(child1, child2)
                  del child1.fitness.values
                  del child2.fitness.values

              # Apply mutation on the offspring
              for mutant in offspring:
                if random.random() < MUTPB:
                  toolbox.mutate(mutant)
                  del mutant.fitness.values

              # Evaluate the individuals with an invalid fitness
              invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
              fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
              for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

              # The population is entirely replaced by the offspring
              pop[:] = offspring
              hall_of_fame.update(pop)
              print(
                "HALL OF FAME:\n"
                + "\n".join(
                  [
                    f"  {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                    for _, ind in enumerate(hall_of_fame)
                  ]
                )
              )

              fitnesses = [
                ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
              ]
              mean[g] = np.mean(fitnesses)
              best[g] = np.max(fitnesses)

            end_t = time.perf_counter()
            print(f"Time Elapsed: {end_t - t:,.2f}")

            # 최적의 파라미터 값 출력
            OPTIMISED_STRATEGY_PARAMS = {
              k: v for k, v in zip(PARAM_NAMES, hall_of_fame[0])}
            STC_params = list(OPTIMISED_STRATEGY_PARAMS.values())
            print('**STC 파라미터 값: ', STC_params)

            

        except:
            STC_params = [14, 3, 3, 80, 20]


        # 데이터 불러와서 가공 ------------------------------------------------------------
        #df = pd.read_json(json_data[fullcode_list0[j]], orient ='index') 
        #read_df = df.transpose()
        
        
        read_df = pd.read_csv(f)
        read_df["GDC_sig"] = ""
        read_df["RSI_sig"] = ""
        read_df["ROC_sig"] = ""
        read_df["STC_sig"] = ""
        read_df["MAP_sig"] = ""

        read_df['TDD_CLSPRC'] = backtest_data(read_df['TDD_CLSPRC'])
        read_df['TDD_HGPRC'] = backtest_data(read_df['TDD_HGPRC'])
        read_df['TDD_LWPRC'] = backtest_data(read_df['TDD_LWPRC'])
        read_df['pfast'] = talib.MA(read_df['TDD_CLSPRC'], timeperiod = GDC_params[0], matype=0)
        read_df['pslow'] = talib.MA(read_df['TDD_CLSPRC'], timeperiod = GDC_params[1], matype=0)
        read_df['RSI'] = talib.RSI(read_df['TDD_CLSPRC'], timeperiod = RSI_params[0])
        read_df['ROC'] = talib.ROC(read_df['TDD_CLSPRC'], timeperiod = ROC_params[0]) 
        read_df['slowk'], read_df['slowd'] = talib.STOCH(read_df['TDD_HGPRC'], read_df['TDD_LWPRC'], read_df['TDD_CLSPRC'], fastk_period = STC_params[0], slowk_period = STC_params[1], slowd_period = STC_params[2], slowk_matype=0, slowd_matype=0)
        read_df['MA'] = talib.MA(read_df['TDD_CLSPRC'], timeperiod = MAP_params[0])
        ul = []
        ll = []
        for i in read_df['MA']:
            uls = i + (MAP_params[1] * i)
            lls = i - (MAP_params[2] * i)
            ul.append(uls)
            ll.append(lls)

        read_df['ul'] = ul
        read_df['ll'] = ll

        # 매도, 매수 전략 설정 후 GDC_sig 열 추가
        first_cross = 0 
        for i in range(0, len(read_df)):
            if read_df['pfast'][i] < read_df['pslow'][i] and first_cross == 0:
              # print('Death cross on day', df['TRD_DD'][i], ':expect the price to continue to fall (매도)')
              read_df['GDC_sig'][i] = 1
              first_cross=1
            elif read_df['pfast'][i] > read_df['pslow'][i] and first_cross ==1:
              # print('Golden cross on day', df['TRD_DD'][i], ':expect the price to continue to rise (매수)')
              first_cross=0
              read_df['GDC_sig'][i] = -1
            else:
              read_df['GDC_sig'][i] = 0

        # 매도 매수 전략 설정 후 RSI_sig 열 추가
        for i in range(0, len(read_df)):
            if read_df['RSI'][i] < 30: # 30보다 작으면 매수시점
                read_df['RSI_sig'][i] = -1
            elif read_df['RSI'][i] >= 70: # 70보다 크면 매도시점
                read_df['RSI_sig'][i] = 1
            else:
                read_df['RSI_sig'][i] = 0

        # 매도 매수 전략 설정 후 ROC_sig 열 추가
        for i in range(0, len(read_df)):
            if read_df['ROC'][i] < 0: # 30보다 작으면 매도시점
                read_df['ROC_sig'][i] = 1
            elif read_df['ROC'][i] >= 0: # 70보다 크면 매수시점
                read_df['ROC_sig'][i] = -1
            else:
                read_df['ROC_sig'][i] = 0      

        # 매도 매수 전략 설정 후 MAP_sig 열 추가
        for i in range(0, len(read_df)):
            if read_df['TDD_CLSPRC'][i] > read_df['ul'][i]: # 매도
              read_df['MAP_sig'][i] = 1
            elif read_df['TDD_CLSPRC'][i] < read_df['ll'][i]: # 매수
              read_df['MAP_sig'][i] = -1
            else:
              read_df['MAP_sig'][i] = 0

        # 매도 매수 전략 설정 후 STC_sig 열 추가
        for i in range(0, len(read_df)):
            if read_df['slowk'][i] < read_df['slowd'][i] and read_df['slowd'][i] < STC_params[4]:
              read_df['STC_sig'][i] = -1
            elif read_df['slowk'][i] > read_df['slowd'][i] and read_df['slowd'][i] > STC_params[3]:
              read_df['STC_sig'][i] = 1
            else:
              read_df['STC_sig'][i] = 0

        # result = read_df.drop(['pfast', 'pslow', 'RSI'], axis='columns')
        result = read_df[['TRD_DD','MKTCAP', 'GDC_sig', 'RSI_sig', 'ROC_sig', 'MAP_sig', 'STC_sig']]        
        #print(result)

        # 조건에 해당하는 날짜 추출.
        def get_point(result):
          x = list(result['x1'])+ list(result['x2'])
          x = list(set(x))
          x.sort()
          return x

        # 전체 df에서 해당 날짜만 가져오기
        def get_date(date_list):
            global scode1
  
            if(len(date_list)==0):   #아예 조건에 해당하는 점이 없을 경우
                return pd.DataFrame() 
    
            check_df =  scode1[scode1.x == date_list[0]]

            for i in date_list :
                df = scode1[scode1.x == i]
                check_df = check_df.append(df,ignore_index = True)

            check_df = check_df.iloc[1:,:]
            return check_df


        # 두 점 사이 관계 df 
        def two_point (check_df):

            df = check_df[['x']]
            df = df.iloc[:-1]

            df['x2']= np.nan
            df['y1']= np.nan
            df['y2']= np.nan
            df['t']= np.nan
            df['p']= np.nan
            df['m']= np.nan
            
            df.rename(columns ={'x':'x1'}, inplace = True)

            for i in range(len(df)): 
                df.iloc[i,1] = check_df.iloc[i+1,0]
                df.iloc[i,2] = check_df.iloc[i,1]
                df.iloc[i,3] = check_df.iloc[i+1,1]

            for i in range(df.shape[0]):

                t = df.iloc[i,1] - df.iloc[i,0]
                df.iloc[i,4] = t.days

                y1 = df.iloc[i,2]
                y2 = df.iloc[i,3]
                result = (abs(y2 - y1 )) / ((y1+y2)/2 )
                df.iloc[i,5] = result

                m = df.iloc[i,3]- df.iloc[i,2]  
                if (m>0):
                    df.iloc[i,6] = 1
                elif (m<0):
                    df.iloc[i,6] = -1
                else :
                    df.iloc[i,6] = 0

            return df


        # 조건 필터링 
        def p_t(df):
            t = df['t'] <5
            p = df['p']<0.05

            result = df[~t&~p]
            return result

        
        
        scode = result.copy()
        
        #시간 순 재정렬.
        scode2 = scode.sort_values(by=['TRD_DD'])
        scode2.reset_index(drop=True,inplace=True)
        scode2['TRD_DD']=pd.to_datetime(scode2['TRD_DD']) #datetime변환

        #시가총액 str->float 데이터타입변환
        scode2['MKTCAP'] = scode2['MKTCAP'].str.replace(',','').astype('float')

        #날짜, 시가총액 열만 추출
        scode1 = scode2[['TRD_DD','MKTCAP']]
        scode1 = scode1.rename(columns = {'TRD_DD':'x','MKTCAP':'y'})
        scode1.reset_index(drop=True,inplace=True)


        #기울기 변하는 지점 찾아주기
        ischange = list()

        for i in range(1,len(scode1)-2):
            m1 = scode1.iloc[i,1] - scode1.iloc[i-1,1]
            m2 = scode1.iloc[i+1,1] - scode1.iloc[i,1]

            if(m1*m2<=0):
                ischange.append(scode1.iloc[i,0])

        # 전체 df에서 기울기 변하는 지점들만 추출한 후 , (t=5, p=0.05) 에 해당하는 날짜  추출.

        check_df = get_date(ischange) #전체 df에서 기울기 변하는 날짜만 추출하기.
        if(len(check_df)==0):

            #scode['TREND']= np.nan
            scode= scode[['TRD_DD', 'GDC_sig', 'RSI_sig', 'ROC_sig', 'MAP_sig', 'STC_sig']]
            
            #result_js = scode.to_json(orient = 'columns')
            #result_dict[j] = result_js
            
            output_path = os.path.join('/opt/ml/processing/processed_data' , code)
            pd.DataFrame(scode).to_csv(output_path, index=False)
            
            print("\n기울기 변하는 날짜가 존재 하지 않음. 조건 성립 X")
            print("-----------------------------------------------------------------------------------------\n")
            print(k,'Saving train data {}'.format(output_path))
            
            continue
        df = two_point(check_df)
        result = p_t(df)   # (t=5, p=0.05) 에 해당하는 날짜  추출.



        # 조건에 해당하는 날짜들끼리 다시 (t=5, p=0.05) 에 해당하는 날짜  추출. 
        red_x = get_point(result)
        red = get_date(red_x)

        if(len(red)==0): #아예 trend 점 안만들어지는 종목 에러 방지.
            
            
            scode= scode[['TRD_DD', 'GDC_sig', 'RSI_sig', 'ROC_sig', 'MAP_sig', 'STC_sig']]
            
            output_path = os.path.join('/opt/ml/processing/processed_data' , code)
            pd.DataFrame(scode).to_csv(output_path, index=False)
            
            #scode['TREND']= np.nan
            #result_js = scode.to_json(orient = 'columns')

            #result_dict[j] = result_js
            print("조건에 해당하는 점이 아예 없음.")
            print("\n-----------------------------------------------------------------------------------------")
            print(k,'Saving train data {}'.format(output_path))
            continue

        df1=two_point(red)
        result2=p_t(df1)


        #점들이 모두 이어지고, 기울기가 계속 변하는 모습 나올때까지 반복 작업.
        while True:

            count = 0     
            red_x1 = get_point(result2) #조건에 만족하는 날짜 추출

            for i in range(result2.shape[0]-1):
                a = result2.iloc[i,1] == result2.iloc[i+1,0]
                b = result2.iloc[i,6]* result2.iloc[i+1,6]== -1

                if(a&b ) :
                    count+=1

            if (count ==  result2.shape[0]-1 ):
                print("조건 성립 완료 \n")
                break

            for i in range(result2.shape[0]-1):

               #i번째 기울기 음수일때
              if (result2.iloc[i,6]== -1):  
                #i+1번째 기울기 양수일 때
                if (result2.iloc[i+1,6] == 1): 
                  #점이 이어져 있지 않으면
                  if (result2.iloc[i,1] != result2.iloc[i+1,0]):
                    if(result2.iloc[i,3]> result2.iloc[i+1,2]):
                       red_x1.remove(result2.iloc[i,1])
                    else :
                      red_x1.remove(result2.iloc[i+1,0]) 

                #i+1번째 기울기 음수일 때
                elif (result2.iloc [i+1,6]== -1):       
                   red_x1.remove(result2.iloc[i,1])


              #i번째 기울기 양수일때
              else :   
                #i+1번째 기울기 양수일 때
                if (result2.iloc[i+1,6] == 1): 
                  #점이 이어져 있지 않으면
                  if (result2.iloc[i,1] != result2.iloc[i+1,0]):
                    red_x1.remove(result2.iloc[i,1])
                    red_x1.remove(result2.iloc[i+1,0]) 
                  #점이 이어져 있으면
                  else :
                    red_x1.remove(result2.iloc[i,1])
                #i+1번째 기울기 음수일 때
                else :
                  #점이 이어져 있지 않으면
                  if (result2.iloc[i,1] != result2.iloc[i+1,0]):
                    if (result2.iloc[i,3]>=result2.iloc[i+1,2]):
                       red_x1.remove(result2.iloc[i+1,0])
                    else:
                      red_x1.remove(result2.iloc[i,1])

            final = get_date(red_x1)
            df1=two_point(final)
            result2=p_t(df1)



        # trend -1~1 사이 값으로 변환.

        final = get_date(red_x1) #최종 기울기 변하는 점 추출.

        for k in range(final.shape[0]-1): #기울기 변하는 곳 1, -1로 값 채워주기
            if(result2.iloc[k,6]== 1):
                final.iloc[k,1] = -1
            else :
                final.iloc[k,1] = 1

        # 마지막 끝 점 (-1,1)해당하는 값으로 채워주기
        n = final.shape[0]-2
        if(final.iloc[n,1]== -1):
            final.iloc[final.shape[0]-1,1] = 1
        else:
            final.iloc[final.shape[0]-1,1] = -1

        # -1~ 1 사이 점 채워주기.(linear interpolation)
        scode_trend = scode1[['x']]
        scode_trend['TREND'] = np.nan

        for i in range(len(final)):
            scode_trend.loc[scode_trend['x']== final.iloc[i,0],'TREND'] = final.iloc[i,1]

        scode_trend =  scode_trend.set_index('x')
        scode_trend = scode_trend[final.iloc[0,0]:final.iloc[len(final)-1,0]].interpolate(method = "time")


        # 마지막으로 원래 데이터에 TREND 열 만들어주어서 합치기.

        scode_trend.reset_index(inplace=True)
        scode_trend = scode_trend.rename(columns = {'x':'TRD_DD'})
        scode['TRD_DD']=pd.to_datetime(scode['TRD_DD'])
        scode = pd.merge(scode, scode_trend, on='TRD_DD', how='left')
        scode['TRD_DD'] = scode['TRD_DD'].astype(str).str.replace('-','/') 
        
        scode= scode[['TRD_DD', 'GDC_sig', 'RSI_sig', 'ROC_sig', 'MAP_sig', 'STC_sig','TREND']]
        #print(scode)

        output_path = os.path.join('/opt/ml/processing/processed_data' , code)
        #output_path = os.path.join('/opt/ml/processing/processed_data' , f)
        pd.DataFrame(scode).to_csv(output_path, index=False)
        print(k,'Saving train data {}'.format(output_path))
    
    

 
