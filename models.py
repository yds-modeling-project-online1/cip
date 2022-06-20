from tabnanny import verbose
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from dataloader import dataload
from utils import saveDataFrame


class Models():
    def __init__(self, train_dataset, test_dataset, trial = 10):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.trial = trial
        self.result = None

    def inference(self):
        test_err = self.test_dataset
        test_x = test_err.iloc[:, 1:] 

        # 예측
        pred_y_list = []
        # kfold 평균
        models, _, _, _ = self.result
        for model in models:
            pred_y = model.predict(test_x)
            pred_y_list.append(pred_y.reshape(-1,1))

        # # kfold 중 auc best
        # models, recalls, precisions, auc_scores = self.result
        # max_idx = np.argmax(auc_scores)
        # pred_y = models[max_idx].predict(test_x)
        # pred_y_list.append(pred_y.reshape(-1,1))
            
        pred_ensemble = np.mean(pred_y_list, axis = 0)
        sample_submission = dataload('sample submission')
        sample_submission['problem'] = pred_ensemble.reshape(-1)

        # save_path = input('추론 결과를 저장할 경로를 입력해주세요 : ')
        save_path = 'submission'
        print("추론 결과를 저장할 경로 : ", save_path)
        saveDataFrame(save_path, sample_submission)
        print("submission 저장 완료")
        return sample_submission

    def f_pr_auc(self, probas_pred, y_true):    # optuna
        p, r, _ = precision_recall_curve(y_true, probas_pred)
        score=auc(r,p) 
        return score

    def k_folding(self, params, select_model):
        models     = []
        recalls    = []
        precisions = []
        auc_scores   = []
        threshold = 0.5

        train_x = self.train_dataset.iloc[:, 1:-1] 
        train_y = self.train_dataset.iloc[:,-1] 
        print("K-fold")
        print(train_x.shape)
        print(train_y.shape)

        # Kfold cross validation
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in k_fold.split(train_x):

            # split train, validation set
            X = train_x.iloc[train_idx]
            y = train_y.iloc[train_idx]
            valid_x = train_x.iloc[val_idx]
            valid_y = train_y.iloc[val_idx]
            
            #run traning
            model = select_model(**params)
            if select_model in [lgb.LGBMRegressor, xgb.XGBRegressor, xgb.XGBRFRegressor]:
                model.fit(X, y, eval_set=[(valid_x, valid_y)], verbose=1)
            else:
                model.fit(X,y)

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

            print('============================ k-fold 실행 중 ============================')
        print(models, recalls, precisions, auc_scores)
        self.result = models, recalls, precisions, auc_scores


    def LR(self):
        params = {}
        self.k_folding(params, LinearRegression)
        return self.result

    def LGBM(self):
        # optuna
        train_x = self.train_dataset.iloc[:, 1:-1] 
        train_y = self.train_dataset.iloc[:,-1] 
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
        sampler = TPESampler(seed=10)

        def objective(trial):
            param = {
                'boosting_type'         : 'gbdt',
                'objective'             : 'regression',
                'seed'                  : 1015,
                'verbose'               : -1,
                'metric'                : 'auc', 
                'max_depth'             : trial.suggest_int('max_depth',5, 20),
                'learning_rate'         : trial.suggest_loguniform("learning_rate", 1e-8, 0.8),
                'num_iterations'        : trial.suggest_int('num_iterations',100,1500),
            }
            
            model = lgb.LGBMRegressor(**param)
            model = model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=25)
            pred = model.predict(valid_x)
            auc = self.f_pr_auc(pred, valid_y)
            return auc
                
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials = self.trial)
        best_params = study.best_params

        self.k_folding(best_params, lgb.LGBMRegressor)
        return self.result

    def XGB(self):
        # optuna
        train_x = self.train_dataset.iloc[:, 1:-1] 
        train_y = self.train_dataset.iloc[:,-1] 
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
        sampler = TPESampler(seed=10)

        def objective(trial):
            param = {
                'learning_rate'     : trial.suggest_loguniform("learning_rate", 1e-8, 0.5),
                'n_estimators'      : trial.suggest_int("n_estimators", 100,1000),
                'max_depth'         : trial.suggest_int("max_depth",3,8),
                'min_child_weight'  : trial.suggest_int("min_child_weight", 1, 10),
                'gamma'             : trial.suggest_int("gamma", 0, 10),
                'subsample'         : trial.suggest_float("subsample", 0.5, 1),
                'eval_metric' : 'auc',
            }

            model = xgb.XGBRegressor(**param)
            model = model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=25)
            pred = model.predict(valid_x)
            auc = self.f_pr_auc(pred, valid_y)
            return auc

        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials = self.trial)
        best_params = study.best_params
        best_params['eval_metric'] = 'auc'

        self.k_folding(best_params, xgb.XGBRegressor)
        return self.result

    def XGBRF(self):
        # optuna
        train_x = self.train_dataset.iloc[:, 1:-1] 
        train_y = self.train_dataset.iloc[:,-1] 
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
        sampler = TPESampler(seed=10)

        def objective(trial):
            param = {
                'max_depth' : trial.suggest_int('max_depth', 5, 15),
                'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
                'num_parallel_tree': trial.suggest_int('num_parallel_tree',5,100),
                'n_estimators': trial.suggest_int('n_estimators',50,100),
                'eval_metric' : 'auc',
            }

            model = xgb.XGBRFRegressor(**param)
            model = model.fit(train_x, train_y)
            pred = model.predict(valid_x)
            auc = self.f_pr_auc(pred, valid_y)
            return auc

        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials = self.trial)
        best_params = study.best_params
        best_params['eval_metric'] = 'auc'

        self.k_folding(best_params, xgb.XGBRFRegressor)
        return self.result

    def RF(self):
        # optuna
        train_x = self.train_dataset.iloc[:, 1:-1] 
        train_y = self.train_dataset.iloc[:,-1] 
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
        sampler = TPESampler(seed=10)

        def objective(trial):

            param = {
                'n_estimators': trial.suggest_int('n_estimators', 15, 100),
                'max_depth' : trial.suggest_int('max_depth', 9, 20),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 3, 15),
                'min_samples_split' : trial.suggest_int('min_samples_split', 5, 20),
            }

            model = RandomForestRegressor(**param)
            rf_model = model.fit(train_x, train_y)
            auc = self.f_pr_auc(rf_model.predict(valid_x),valid_y)
            return auc

        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials = self.trial)
        best_params = study.best_params

        self.k_folding(best_params, RandomForestRegressor)
        return self.result