import datetime as dt # Analysis
from tqdm import tqdm


class preprocess():
    def __init__(self, train_error, train_quality, train_problem, test_err, test_quality):
        self.train_err = train_error
        self.train_qual = train_quality
        self.train_prob = train_problem
        self.test_err = test_err
        self.test_qual = test_quality

    def str2int(x):
        if type(x) == str:
            x = x.replace(",","")
            x = int(x)
            return x
        else:
            x = int(x)
            return x

    def make_datetime(x): # datetime 데이터로 변환
        x = str(x)
        year  = int(x[:4])
        month = int(x[4:6])
        day   = int(x[6:8])
        hour  = int(x[8:10])
        minute  = int(x[10:12])
        sec  = int(x[12:])
        return dt.datetime(year, month, day, hour, minute, sec)

    def missingValue(self): # 결측치 처리
        # train_err
        self.train_err['errcode'] = self.train_err['errcode'].fillna('40013')
        
        # train_qual
        for i in tqdm(self.train_qual.columns[3:]):
            self.train_qual[i] = self.train_qual[i].fillna(self.train_qual[i].mode(0)[0])  # mode : 해당 시리즈의 가장 높은 빈도 값을 반환
        fwms_idx = self.train_qual[self.train_qual['fwver'].isnull()].index
        self.train_qual = self.train_qual.drop(fwms_idx)

        # test_err
        for i in range(len(self.test_err[self.test_err.errcode.isnull()])):
            idx = self.test_err[self.test_err.errcode.isnull()].index[i]
            u, t, m, f, e, _ = self.test_err[self.test_err.errcode.isnull()].iloc[i]
            self.test_err.errcode.iloc[idx] = (self.test_err[(self.test_err.user_id == u) & (self.test_err.time == t) & (self.test_err.fwver == f)\
                & (self.test_err.errtype == e)].errcode.mode())

        # test_qual
        fwms_idx = self.test_qual[self.test_qual['fwver'].isnull()].index
        self.test_qual = self.test_qual.drop(fwms_idx)
        for i in self.test_qual.columns[3:]:
            self.test_qual[i] = self.test_qual[i].fillna(self.test_qual[i].mode(0)[0])

    def modifyType(self):   # 각 column들을 적절한 type으로 변환
        for i in self.train_qual.columns[3:]:
            self.train_qual[i] = self.train_qual[i].apply(lambda x : self.str2int(x))
        for i in self.test_qual.columns[3:]:
            self.test_qual[i] = self.test_qual[i].apply(lambda x : self.str2int(x)) 
        
        self.train_err.time = self.train_err.time.apply(lambda x : self.make_datetime(x))
        self.train_qual = self.train_qual.sort_values(['user_id','time']).reset_index(drop=True)
        self.train_qual.time = self.train_qual.time.apply(lambda x : self.make_datetime(x))
        self.train_prob = self.train_prob.sort_values(['user_id','time']).reset_index(drop=True)
        self.train_prob.time = self.train_prob.time.apply(lambda x : self.make_datetime(x))
        self.test_err.time = self.test_err.time.apply(lambda x : self.make_datetime(x))
        self.test_qual.time = self.test_qual.time.apply(lambda x : self.make_datetime(x))

    def saveDataframe(self):
        save_path = input("저장할 경로를 입력하세요 : ")
        print("저장 경로 : ", save_path)

        self.train_err.to_csv(save_path + 'train_err_data_rev.csv', index=False)
        self.train_qual.to_csv(save_path + 'train_quality_data_rev.csv', index=False)
        self.test_err.to_csv(save_path + 'test_err_data_rev.csv', index=False)
        self.test_qual.to_csv(save_path + 'test_quality_data_rev.csv', index=False)
        self.train_prob.to_csv(save_path + 'train_problem_data_rev.csv', index=False)
        print("저장 완료")