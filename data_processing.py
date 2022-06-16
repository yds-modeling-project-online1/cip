import numpy as np # Analysis
import pandas as pd # Analysis

from tqdm import tqdm

from dataloader import dataload


class cipDataset():
    def __init__(self, train_dataset, test_dataset, data_path='./data/', save_path='./processing_result/'):
        self.data_path = data_path          # train, test, problem 데이터 경로
        self.save_path = save_path                  # 저장 경로
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def labeling(self, train_df, target_df):
        problem = np.zeros(train_df.user_id.nunique())
        # error와 동일한 방법으로 person_idx - 10000 위치에 
        # person_idx의 problem이 한 번이라도 발생했다면 1
        # 없다면 0
        problem[target_df.user_id.unique() - train_df.user_id.min()] = 1
        return problem

    def makeFeatures(self, df, flag='train'): # flag : train 데이터라면 'train', test 데이터라면 'test'

        # -------------------- time column drop하기 --------------------

        df_rev1 = df.drop('time', axis=1)

        # -------------------- time column drop하기 끝 --------------------


        # -------------------- 사용자별 데이터로 구축 (별도의 메서드로 구현해주면 좋을듯) --------------------

        # train 데이터 설명을 확인하면 user_id가 10000부터 24999까지 총 15000개가 연속적으로 존재.
        # test 데이터는 user_id가 30000부터 44998까지 총 14999개가 연속적으로 존재. (그런데 데이터에는 중간에 한 id(43262)는 존재하지 않음)
        user_id_max = df.user_id.max()
        user_id_min = df.user_id.min()
        user_number = user_id_max - user_id_min + 1

        # -------------------- 사용자별 errtype feature 추가 --------------------
        id_error = df_rev1[['user_id','errtype']].values
        id_errors ={}
        for i in range(user_id_min, user_id_max + 1):
            id_errors[i] = {}
        for u_id, err in tqdm(id_error):
            if err not in id_errors[u_id]:
                id_errors[u_id][err] = 0
            id_errors[u_id][err] += 1

        df_rev2 = pd.DataFrame({'user_id' : pd.Series(id_errors.keys()), 'errtype' : pd.Series(id_errors.values())})

        # 42 : errtype이 1 ~ 42까지 존재
        error = np.zeros((user_number, 42)) 

        for i, (key, value) in tqdm(enumerate(df_rev2.values), total=len(df_rev2)):
            for k in value.keys():
                # test 데이터에는 user_id가 30000 ~ 44998까지 있는데 고유 고객 수는 14998이라서 index 값으로 series가 구성되도록 수정했음
                # key - train_user_id_min, errtype에 사용자가 접한 errtype 빈도 값을 저장
                error[i, k - 1] += value[k]

        col_errtypes = []
        for i in range(1, 42+1):
            col_errtypes.append(f"errtype{i}")
        
        df_rev2 = pd.concat([df_rev2, pd.DataFrame(error.astype(int), columns=col_errtypes)], axis=1).drop('errtype', axis=1)

        # -------------------- 사용자별 errtype feature 추가 끝 --------------------


        # -------------------- 사용자별 모델 변경 횟수 feature 추가 (별도의 메서드로 구현해주면 좋을듯) --------------------

        # 모델 2개 이상에서 오류가 발생한 사용자 리스트
        alt_models_idx = df_rev1.groupby('user_id').nunique()[df_rev1.groupby('user_id').nunique().model_nm > 1].index
        # 모델 변경 경험이 있는 사용자의 id와 변경 횟수 dictionary
        alt_models_dict = df_rev1[df_rev1.user_id.isin(alt_models_idx)].groupby('user_id').model_nm.nunique().to_dict()

        num_models = np.zeros(user_number)
        for key, value in tqdm(list(alt_models_dict.items())):
                num_models[key - user_id_min] = value
        df_rev2 = pd.concat([df_rev2, pd.DataFrame(num_models.astype(int), columns=['alt_model'])], axis=1)

        # -------------------- 사용자별 모델 변경 횟수 feature 추가 끝 --------------------

        # -------------------- 사용자별 펌웨어 버전 변경 횟수 feature 추가 (별도의 메서드로 구현해주면 좋을듯) --------------------

        # 펌웨어버전 2개 이상에서 오류가 발생한 사용자 리스트
        alt_fwvers_idx = df_rev1.groupby('user_id').nunique()[df_rev1.groupby('user_id').nunique().fwver > 1].index
        # 펌웨어버전 변경 경험이 있는 사용자의 id와 변경 횟수 dictionary
        alt_fwvers_dict = df_rev1[df_rev1.user_id.isin(alt_fwvers_idx)].groupby('user_id').fwver.nunique().to_dict()

        num_fwvers = np.zeros(user_number)
        for key, value in tqdm(list(alt_fwvers_dict.items())):
            num_fwvers[key - user_id_min] = value
        df_rev2 = pd.concat([df_rev2, pd.DataFrame(num_fwvers.astype(int), columns=['alt_fwver'])], axis=1)

        # -------------------- 사용자별 펌웨어 버전 변경 횟수 feature 추가 끝 --------------------

        # -------------------- 사용자별 데이터로 구축 끝 --------------------


        # -------------------- target column 추가해주기 --------------------

        if flag == 'train':
            target_filename = dataload('problem 데이터를 선택해주세요')
            target_df = pd.read_csv(target_filename)
            target = self.labeling(df_rev2, target_df)
            df_rev2 = pd.concat([df_rev2, pd.DataFrame(target.astype(int), columns=['target'])], axis=1)

        # -------------------- target column 추가해주기 끝 --------------------

        return df_rev2

    