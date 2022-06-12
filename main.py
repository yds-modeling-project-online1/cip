import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument(
        '-load_path', help='insert data path and filename, dir/filename', default=None, type=str)
    parser.add_argument(
        '-save_path', help='insert save path and filename, dir/filename', default=None, type=str)
    parser.add_argument(
        '-flag', help='select data to process, "train" or "test"', default=None, type=str)
    parser.add_argument(
        '-logs', help='insert logs path and filename, dir/filename', default=None, type=str)
    args = parser.parse_args()

    # ex) load_path = 'cip/EDA/preprocessing/'           # 데이터 불러오는 경로
    # ex) save_path = 'cip/EDA/processing_result/'       # 데이터 처리 후 데이터 저장하는 경로