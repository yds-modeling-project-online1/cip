from data_processing import cipDataset
from preprocess import preprocess
from dataloader import dataload
from utils import saveDataFrame
from models import Models
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument(
        '-load_path', help='insert data path and filename, dir/filename', default='./data/', type=str)
    parser.add_argument(
        '-save_path', help='insert save path and filename, dir/filename', default='./processing_result/', type=str)
    parser.add_argument(
        '-flag', help='select preprocess to train or build data to train or just train, "0" or "1" or "2"', default=0, type=int)
    parser.add_argument(
        '-logs', help='insert logs path and filename, dir/filename', default='./logs/', type=str)
    args = parser.parse_args()

    # ex) load_path = 'cip/EDA/preprocessing/'           # 데이터 불러오는 경로
    # ex) save_path = 'cip/EDA/processing_result/'       # 데이터 처리 후 데이터 저장하는 경로

    if args.flag == 1:
        train_err = dataload("train error 데이터")
        train_qual = dataload("train quality 데이터")
        train_prob = dataload("train problem 데이터")
        test_err = dataload("test error 데이터")
        test_qual = dataload("test quality 데이터")

        preproc = preprocess(train_err, train_qual, train_prob, test_err, test_qual)

    if args.flag <= 1:
        train_data = dataload('전처리된 train error 데이터')
        test_data = dataload('전처리된 test error 데이터')
        data = cipDataset(train_data, test_data, args.load_path, args.save_path)
        train_dataset = data.makeFeatures(train_data, 'train')
        test_dataset = data.makeFeatures(test_data, 'test')
        saveDataFrame(args.save_path, train_dataset)
        saveDataFrame(args.save_path, test_dataset)

        model = Models(train_dataset, test_dataset)
        print(model.LGBM())
        model.inference()

    else:
        train_dataset = dataload('train 데이터셋')
        test_dataset = dataload('test 데이터셋')
        
        model = Models(train_dataset, test_dataset)
        print(model.LGBM())
        model.inference()