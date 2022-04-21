
from cgi import test
import pandas as pd
import random

def create_csv(file, len_, file_name):
    max_col = file.columns.size
    ancor = {}
    to_return = []
    while len(to_return) < len_:
        toadd = []
        col1 = random.randint(0, max_col - 1)
        col2 = -1
        while 1:
            col2 = random.randint(0, max_col -1)
            if col2 != col1:
                break
        max_row1 = file.iat[0, col1]
        max_row2 = file.iat[0, col2]
        row1 = random.randint(1, max_row1)
        row2 = random.randint(1, max_row2)
        ancor_row = ""
        if  col1 not in ancor:
            ancor_row = random.randint(1, max_row1)
            ancor[col1] = ancor_row
        else:
            ancor_row = ancor[col1]
        toadd.append(file.iat[ancor_row, col1])
        toadd.append(file.iat[row1, col1])
        toadd.append(file.iat[row2, col2])
        counter = to_return.count(toadd)
        if counter == 0:
            to_return.append(toadd)
    csv_file = pd.DataFrame(to_return)
    headerr = ["Ancor","positive", "Negative"]
    csv_file = csv_file.sample(frac=1)
    csv_file.to_csv(file_name,
                        index=False, sep=',', header=headerr)


if __name__ == "__main__":

    train_file  = pd.read_excel('1-230Arabic.xlsx')
    test_file = pd.read_excel('50D_test_arabic.xlsx')
    create_csv(train_file, 24000, 'train_triplet.csv')
    create_csv(test_file, 8000, 'test_triplet.csv')
