import pandas as pd

def saveDataFrame(save_path, df):
    save_filename = input("저장할 파일 이름을 입력하세요(확장자명도 포함) : ")  # 키보드 입력 대신 GUI로 경로 선택할 수 있게 만들어주면 좋을듯
    save_path = save_path + '/' + save_filename
    print("저장 경로 : ", save_path)

    df.to_csv(save_path, index=False)
    print("저장 완료")