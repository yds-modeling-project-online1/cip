import numpy as np # Analysis
import pandas as pd # Analysis

from tqdm import tqdm

class cipDataset():
    def __init__(self, data_path, save_path):
        self.data_path = data_path          # train, test, problem 데이터 경로
        # self.target_path = target_path      # problem 데이터 경로
        self.save_path = save_path                  # 저장 경로

    def labeling(self, train_filename):
        target_filename = input("target 데이터 파일 이름을 입력해주세요 : ")
        print("train 데이터 경로 : ", self.data_path + train_filename)
        print("target 데이터 경로 : ", self.data_path + target_filename)

        train_data = pd.read_csv(self.data_path + train_filename)
        target_data = pd.read_csv(self.data_path + target_filename)
        problem = np.zeros(train_data.user_id.nunique())
        # error와 동일한 방법으로 person_idx - 10000 위치에 
        # person_idx의 problem이 한 번이라도 발생했다면 1
        # 없다면 0
        problem[target_data.user_id.unique() - train_data.user_id.min()] = 1
        return problem

    def makeFeatures(self, flag):   # flag : train 데이터인지 test 데이터인지 선택
        data_filename = input("데이터 파일 이름을 입력해주세요 : ")
        print("train 데이터 경로 : ", self.data_path + data_filename)

        try:
            error_df = pd.read_csv(self.data_path + data_filename)
        except:
            print('데이터 경로가 올바르지 않습니다. 데이터 경로를 다시 확인해주세요.')
            return

        # -------------------- time column drop하기 --------------------

        error_df_rev1 = error_df.drop('time', axis=1)

        # -------------------- time column drop하기 끝 --------------------



        # -------------------- 펌웨어 버전 변경 유무, 모델 변경 유무 feature 만들기 (별도의 메소드로 구현해주면 좋을듯) --------------------

        # 펌웨어버전 2개 이상에서 오류가 발생한 사용자 리스트
        alt_fwvers_idx = error_df_rev1.groupby('user_id').nunique()[error_df_rev1.groupby('user_id').nunique().fwver > 1].index

        # 모델 2개 이상에서 오류가 발생한 사용자 리스트
        alt_models_idx = error_df_rev1.groupby('user_id').nunique()[error_df_rev1.groupby('user_id').nunique().model_nm > 1].index

        alt_fwvers = np.zeros(len(error_df_rev1), dtype=int)
        alt_models = np.zeros(len(error_df_rev1), dtype=int)

        error_df_rev1['alt_fwver'] = alt_fwvers
        error_df_rev1['alt_model'] = alt_models

        for idx in tqdm(error_df_rev1[error_df_rev1.user_id.isin(alt_fwvers_idx)].index):
            alt_fwvers[idx] = 1

        for idx in tqdm(error_df_rev1[error_df_rev1.user_id.isin(alt_models_idx)].index):
            alt_models[idx] = 1

        error_df_rev1['alt_fwver'] = alt_fwvers
        error_df_rev1['alt_model'] = alt_models

        # -------------------- 펌웨어 버전 변경 유무, 모델 변경 유무 feature 만들기 끝 --------------------



        # -------------------- 사용자별 데이터로 구축 (별도의 메소드로 구현해주면 좋을듯) --------------------

        # train 데이터 설명을 확인하면 user_id가 10000부터 24999까지 총 15000개가 연속적으로 존재.
        user_id_max = error_df.user_id.max()
        user_id_min = error_df.user_id.min()
        user_number = error_df.user_id.nunique()

        id_error = error_df_rev1[['user_id','errtype']].values
        id_errors = {}
        for u_id, err in tqdm(id_error):
            if u_id not in id_errors:
                id_errors[u_id] = {}
            if err not in id_errors[u_id]:
                id_errors[u_id][err] = 0
            id_errors[u_id][err] += 1

        error_df_rev2 = pd.DataFrame({'user_id' : pd.Series(id_errors.keys()), 'errtype' : pd.Series(id_errors.values())})

        # 42 : errtype이 1 ~ 42까지 존재
        error = np.zeros((user_number, 42)) 

        for i, (key, value) in tqdm(enumerate(error_df_rev2.values), total=len(error_df_rev2)):
            for k in value.keys():
                # test 데이터에는 user_id가 30000 ~ 44998까지 있는데 고유 고객 수는 14998이라서 index 값으로 series가 구성되도록 수정했음
                # key - train_user_id_min, errtype에 사용자가 접한 errtype 빈도 값을 저장
                error[i, k - 1] += value[k]

        col_errtypes = []
        for i in range(1, 42+1):
            col_errtypes.append(f"errtype{i}")
        
        error_df_rev2 = pd.concat([error_df_rev2, pd.DataFrame(error.astype(int), columns=col_errtypes)], axis=1).drop('errtype', axis=1)

        num_models = np.zeros(user_number)
        for i in tqdm(list(error_df_rev1[error_df_rev1.alt_model != 0].user_id.unique())):
            num_models[i - user_id_min] = 1
        error_df_rev2 = pd.concat([error_df_rev2, pd.DataFrame(num_models.astype(int), columns=['alt_model'])], axis=1)

        num_fwvers = np.zeros(user_number)
        for i in tqdm(list(error_df_rev1[error_df_rev1.alt_fwver != 0].user_id.unique())):
            num_fwvers[i - user_id_min] = 1
        error_df_rev2 = pd.concat([error_df_rev2, pd.DataFrame(num_fwvers.astype(int), columns=['alt_fwver'])], axis=1)

        # -------------------- 사용자별 데이터로 구축 끝 --------------------


        # -------------------- target column 추가해주기 --------------------

        # train이라고 했지만 target 데이터와 일치하지 않는 데이터의 경우에 대해서 예외처리 필요
        if flag == 'train':
            target = self.labeling(data_filename)
            error_df_rev2 = pd.concat([error_df_rev2, pd.DataFrame(target.astype(int), columns=['target'])], axis=1)

        # -------------------- target column 추가해주기 끝 --------------------

        return error_df_rev2

    def saveDataFrame(self, df):
        save_filename = input("저장할 파일 이름을 입력하세요(확장자명도 포함) : ")
        save_path = self.save_path + save_filename
        print("저장 경로 : ", save_path)

        df.to_csv(save_path, index=False)
        print("저장 완료")