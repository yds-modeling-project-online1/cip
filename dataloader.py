import pandas as pd
from tkinter import filedialog, messagebox

def dataload(msg):
    file = filedialog.askopenfilename(initialdir="/", 
                                      title = msg + " 파일을 선택해주세요", 
                                      filetypes= (("*.csv","*csv"),("*.xlsx","*xlsx"),("*.xls","*xls")),
                                      )

    if file == '':
        messagebox.showwarning("경고", "파일을 선택해주세요")

    df = pd.read_scv(file)
    return df