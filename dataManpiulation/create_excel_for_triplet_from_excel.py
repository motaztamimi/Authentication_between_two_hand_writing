
import pandas as pd
import random
from prepare_data import create_label_file
def create_miss_match_csv_triplet(file, len_, file_name):
    # must the ancor be from a  and positive from b writer 
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
        while "b" not in file.iat[row1, col1]:
            row1 = random.randint(1, max_row1)


        row2 = random.randint(1, max_row2)
        ancor_row = ""
        if  col1 not in ancor:
            ancor_row = random.randint(1, max_row1)
            while "a" not in file.iat[ancor_row, col1]:
                ancor_row = random.randint(1, max_row1)
            ancor[col1] = ancor_row
        else:
            ancor_row = ancor[col1]
        toadd.append(file.iat[ancor_row, col1])
        toadd.append(file.iat[row1, col1])
        toadd.append(file.iat[row2, col2])
        toadd.append(1)
        counter = to_return.count(toadd)
        if counter == 0:
            to_return.append(toadd)
    csv_file = pd.DataFrame(to_return)
    # headerr = ["Ancor","positive", "Negative", "Label"]
    csv_file = csv_file.sample(frac=1)
    csv_file.to_csv(file_name,
                        index=False, sep=',')

def create_match_csv_triplet(file, len_, file_name):
    max_col = file.columns.size
    ancor = {}
    to_return = []
    while len(to_return) < len_:
        toadd = []
        col1 = random.randint(0, max_col - 1)
        max_row1 = file.iat[0, col1]
        ancor_row = ""
        if  col1 not in ancor:
            ancor_row = random.randint(1, max_row1)
            while "a" not in file.iat[ancor_row, col1]:
                ancor_row = random.randint(1, max_row1)
            ancor[col1] = ancor_row
        else:
            ancor_row = ancor[col1]
        
        row1 = random.randint(1, max_row1)
        while row1 == ancor_row or "b" not in file.iat[row1, col1]:
            row1 = random.randint(1, max_row1)
        row2 = random.randint(1, max_row1)
        while row1 == row2 or row2 == ancor_row:
            row2 = random.randint(1, max_row1)

        toadd.append(file.iat[ancor_row, col1])
        toadd.append(file.iat[row1, col1])
        toadd.append(file.iat[row2, col1])
        toadd.append(0)
        counter = to_return.count(toadd)
        if counter == 0:
            to_return.append(toadd)
    csv_file = pd.DataFrame(to_return)
    # headerr = ["Ancor","positive", "Negative", "Label"]
    csv_file = csv_file.sample(frac=1)
    csv_file.to_csv(file_name,
                        index=False, sep=',')



def find_miss_match_for_triplet_hebrew(file, len_, file_name):
      # must the ancor be from a  and positive from b writer 
    max_col = file.columns.size
    print("hi")
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
    
        ancor_row = ""
        if  col1 not in ancor:
            ancor_row = random.randint(1, max_row1)
            ancor[col1] = ancor_row
        else:
            ancor_row = ancor[col1]

        row1 = random.randint(1, max_row1)
        while row1 == ancor_row:
            row1 = random.randint(1, max_row1)
        row2 = random.randint(1, max_row2)
        toadd.append(file.iat[ancor_row, col1])
        toadd.append(file.iat[row1, col1])
        toadd.append(file.iat[row2, col2])
        toadd.append(1)
        counter = to_return.count(toadd)
        if counter == 0:
            to_return.append(toadd)
    csv_file = pd.DataFrame(to_return)
    # headerr = ["Ancor","positive", "Negative", "Label"]
    csv_file = csv_file.sample(frac=1)
    csv_file.to_csv(file_name,
                        index=False, sep=',')

def find_match_for_triplet_hebrew(file, len_, file_name):
    max_col = file.columns.size
    ancor = {}
    to_return = []
    while len(to_return) < len_:
        toadd = []
        col1 = random.randint(0, max_col - 1)
        max_row1 = file.iat[0, col1]    
        ancor_row = ""
        if  col1 not in ancor:
            ancor_row = random.randint(1, max_row1)
            ancor[col1] = ancor_row
        else:
            ancor_row = ancor[col1]

        row1 = random.randint(1, max_row1)
        while row1 == ancor_row:
            row1 = random.randint(1, max_row1)

        row2 = random.randint(1, max_row1)
        while row1 == row2 or row2 == ancor_row:
            row2 = random.randint(1, max_row1)
        
        toadd.append(file.iat[ancor_row, col1])
        toadd.append(file.iat[row1, col1])
        toadd.append(file.iat[row2, col1])
        toadd.append(0)
        counter = to_return.count(toadd)
        if counter == 0:
            to_return.append(toadd)
    csv_file = pd.DataFrame(to_return)
    # headerr = ["Ancor","positive", "Negative", "Label"]
    csv_file = csv_file.sample(frac=1)
    csv_file.to_csv(file_name,
                        index=False, sep=',')


if __name__ == "__main__":

    # train_file  = pd.read_excel('../1-230Arabic.xlsx')
    # test_file = pd.read_excel('../50D_test_arabic.xlsx')
    train_file = pd.read_excel('../hebrew_data.xlsx',sheet_name='train')
    test_file = pd.read_excel('../hebrew_data.xlsx',sheet_name='test')
    # create_miss_match_csv_triplet(train_file, 25000, '../train_labels_for_arabic_triplet.csv')
    # create_miss_match_csv_triplet(test_file, 8000, '../test_labels_for_arabic_triplet.csv')


    find_miss_match_for_triplet_hebrew(train_file, 12500, '../train_labels_for_hebrew_triplet.csv')

    find_miss_match_for_triplet_hebrew(test_file, 4000, '../test_labels_miss_match_for_hebrew_triplet.csv')
    find_match_for_triplet_hebrew(test_file, 4000, '../test_labels_match_for_hebrew_triplet.csv')

    # # create_match_csv_triplet(test_file, 4000, '../test_labels_match_for_arabic_triplet.csv')
    create_label_file('../test_labels_miss_match_for_hebrew_triplet.csv', '../test_labels_match_for_hebrew_triplet.csv', 4000, '../test_labels_for_hebrew_triplet.csv')