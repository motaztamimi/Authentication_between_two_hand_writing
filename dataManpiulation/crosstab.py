from os import path
import pandas as pd

if __name__ == '__main__':
    excel_file = pd.read_excel(r"C:\Users\FinalProject\Desktop\Motaz_test.xlsx",sheet_name="a")
    max_row = excel_file.shape[0]
    a =pd.DataFrame.groupby(excel_file,by=excel_file['number'])
    excell ={}
    for i in range(max_row):
        if excell.get(excel_file['number'][i]):
            excell[excel_file['number'][i]].append(excel_file['path'][i])
            
        else:
            to_add=[]
            to_add.append(excel_file['path'][i])
            excell[excel_file['number'][i]] =to_add
    main = pd.DataFrame.from_dict(excell,orient='index')
    main=main.transpose()
    main.to_excel('crosstab.xlsx')