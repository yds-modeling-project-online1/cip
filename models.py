import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from utils import saveDataFrame
from dataloader import dataload

# 필요한 함수 정의
def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x     = str(x)
    year  = int(x[:4])
    month = int(x[4:6])
    day   = int(x[6:8])
    hour  = int(x[8:10])
    #mim  = int(x[10:12])
    #sec  = int(x[12:])
    return dt.datetime(year, month, day, hour)

def string2num(x):
    # (,)( )과 같은 불필요한 데이터 정제
    x = re.sub(r"[^0-9]+", '', str(x))
    if x =='':
        return 0
    else:
        return int(x)


import lightgbm as lgb
from sklearn.model_selection import train_test_split


class Models():
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def inference(self):
        test_err = self.test_dataset
        test_x = test_err.iloc[:, 1:] 

        # 예측
        pred_y_list = []
        models, _, _, _ = self.result
        for model in models:
            pred_y = model.predict(test_x)
            pred_y_list.append(pred_y.reshape(-1,1))
            
        pred_ensemble = np.mean(pred_y_list, axis = 0)
        sample_submssion_filename = dataload('sample submission 파일을 선택해주세요')
        sample_submssion = pd.read_csv(sample_submssion_filename)
        sample_submssion['problem'] = pred_ensemble.reshape(-1)

        save_path = input('추론 결과를 저장할 경로를 입력해주세요 : ')
        saveDataFrame(save_path, sample_submssion)
        return sample_submssion

    
    def LGBM(self):
        # 초기화
        train_x = self.dataset.iloc[:, 1:-1] 
        train_y = self.dataset.iloc[:,-1] 
        print(train_x.shape)
        print(train_y.shape)

        # optuna
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)

        sampler = TPESampler(seed=10)

        def f_pr_auc(probas_pred, y_true):
            # labels = y_true.get_label()
            p, r, _ = precision_recall_curve(y_true, probas_pred)
            score=auc(r,p) 
            return score

        def objective(trial):

            param = {
                'objective': 'regression', # 회귀
                'verbose': -1,
                'metric': 'auc', 
                'max_depth': trial.suggest_int('max_depth',5, 20),
                'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 0.8),
                'num_boost_round': trial.suggest_int('num_boost_round',100,1500)
                # 'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                # 'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
            }

            model = lgb.LGBMRegressor(**param)
            lgb_model = model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=25)
            auc = f_pr_auc(lgb_model.predict(valid_x),valid_y)
            return auc
                
        study_lgb = optuna.create_study(direction='maximize', sampler=sampler)
        study_lgb.optimize(objective, n_trials=100)

        best_params = study_lgb.best_params

        # cross validation 전 초기화
        train_x = self.dataset.iloc[:, 1:-1] 
        train_y = self.dataset.iloc[:,-1] 
        print("K-fold")
        print(train_x.shape)
        print(train_y.shape)

        # Train
        #-------------------------------------------------------------------------------------
        # validation auc score를 확인하기 위해 정의
        def f_pr_auc(probas_pred, y_true):
            labels=y_true.get_label()
            p, r, _ = precision_recall_curve(labels, probas_pred)
            score=auc(r,p) 
            return "pr_auc", score, True
        #-------------------------------------------------------------------------------------
        models     = []
        recalls    = []
        precisions = []
        auc_scores   = []
        threshold = 0.5
        # 파라미터 설정
        params =      {
                        'boosting_type'     : 'gbdt',
                        'objective'         : 'regression',
                        'metric'            : 'auc',
                        'seed'              : 1015,
                        'learning_rate'     : best_params['learning_rate'],
                        'max_depth'         : best_params['max_depth'],
                        'num_boost_round'   : best_params['num_boost_round']
                    # ETC
                        # 'min_child_samples' : best_params['min_child_samples'],
                        # 'n_estimators'      : best_params['n_estimators'],
                        # 'subsample'         : best_params['subsample']
                        }
        #-------------------------------------------------------------------------------------
        # 5 Kfold cross validation
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in k_fold.split(train_x):

            # split train, validation set
            X = train_x.iloc[train_idx]
            y = train_y.iloc[train_idx]
            valid_x = train_x.iloc[val_idx]
            valid_y = train_y.iloc[val_idx]

            d_train= lgb.Dataset(X, y)
            d_val  = lgb.Dataset(valid_x, valid_y)
            
            #run traning
            model = lgb.train(
                                params,
                                train_set       = d_train,
                                num_boost_round = 1000,
                                valid_sets      = d_val,
                                feval           = f_pr_auc,
                                verbose_eval    = 20, 
                                early_stopping_rounds = 3,
                            )
            # cal valid prediction
            valid_prob = model.predict(valid_x)
            valid_pred = np.where(valid_prob > threshold, 1, 0)
            
            # cal scores
            recall    = recall_score(    valid_y, valid_pred)
            precision = precision_score( valid_y, valid_pred)
            auc_score = roc_auc_score(   valid_y, valid_prob)

            # append scores
            models.append(model)
            recalls.append(recall)
            precisions.append(precision)
            auc_scores.append(auc_score)

            print('==========================================================')

        self.result = models, recalls, precisions, auc_scores
        return models, recalls, precisions, auc_scores