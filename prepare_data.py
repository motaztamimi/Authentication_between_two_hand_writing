import detection_function
import cv2
import prepare_doc
from PIL import Image
import os
import numpy as np
import pandas as pd
import random
import shutil



def new_func(filename = "match_labels.csv"):
    excel_file=[]
    csv_file = pd.DataFrame(excel_file)
    arr=[]
    arr_not_to_match = []
    for i in range(15):
        person1 , person2= chose_writer(array=arr)
        print(person1)
        arr.append(person1);    
        copy_two_writer_to_another_directory(writer1=person1, writer2=person2)
        a_b = random.randint(0,1)
        typee="a"
        if a_b == 1:
            typee="b" 
        match_csv = find_match_pairs_two_writer(arr_to_ignore = arr_not_to_match ,person=person1,kind=typee)
        mis_match_csv = find_miss_match_pairs_two_writer(person1=person1, person2=person2,kind=typee)
        arr_not_to_match.append(person1)
        csv_file = pd.concat([csv_file,match_csv])
        csv_file = pd.concat([csv_file,mis_match_csv])
        shutil.rmtree("data_for_two_person")
    csv_file.to_csv(filename,index=False, sep=',', header=0)

def chose_writer(array):

    person1=random.randint(65,80)
    person2=random.randint(65,80)
    while person1 in array:
        person1=random.randint(65,80)

    while person1 == person2:
        person2 = random.randint(65,80)    
    return person1 , person2

def copy_two_writer_to_another_directory(path='data_for_each_person', writer1 = 1, writer2 = 2):
    if os.path.exists(path):
        dir_writer1= "person_{}".format(writer1); 
        dir_writer2= "person_{}".format(writer2);
        files_writer1_path = path + '/' + dir_writer1
        files_writer2_path = path + '/' + dir_writer2

        new_path = 'data_for_two_person/' + dir_writer1  
        shutil.copytree(files_writer1_path, new_path)
        new_path1 = 'data_for_two_person/' + dir_writer2 
        shutil.copytree(files_writer2_path, new_path1)


def find_match_pairs_two_writer(path = "data_for_two_person" , filename = "match_labels.csv" , arr_to_ignore = [] ,person =1,kind = "a"):
    
    if os.path.exists(path):
        print('Creating all possible pairs of lines that creat a Match and store the labels in csv file...')
        dir_name = path
        genuin_data = []
        dirs = os.listdir(dir_name)
        dirs= sorted(dirs,key= Sorting_Dir)
        for _dir in dirs:
            if _dir != '.DS_Store':
                files = os.listdir(dir_name + '/' + _dir)
                file =[]
                writer_num = _dir.split("_")[1]
                if int(writer_num) == person  and int(writer_num) not in arr_to_ignore:
                    for i in range(0,5):
                        row=random.randint(1,30)
                        file_name = "p{}_{}_L_{}.jpeg".format(writer_num, kind, row)
                        file_to_search= dir_name + '/' + _dir + '/' + file_name
                        while(os.path.exists(file_to_search) == False):
                            row=random.randint(1,30)
                            file_name = "p{}_{}_L_{}.jpeg".format(writer_num, kind, row)
                            file_to_search= dir_name + '/' + _dir + '/' + file_name
                        file.append(file_name)
                    for indx, img1 in enumerate(file):
                        for img2 in file[indx:]:
                            to_add = []
                            if img1 != img2:
                                to_add.append(_dir + '/' + img1)
                                to_add.append(_dir + '/' + img2)
                                to_add.append('0')
                                genuin_data.append(to_add)
        csv_file = pd.DataFrame(genuin_data)
        csv_file=csv_file.sample(frac=1)
        csv_file= csv_file[0:8]
        print('Done.')
        return csv_file

def find_miss_match_pairs_two_writer(path='data_for_two_person',person1 = 1, person2 = 2, kind = "a" ):
    
    if os.path.exists(path):

        print('Creating all pairs of lines that creat a Miss Match and store the labels in csv file...')
        diff_data = []
        dir_name = path
        dirs = os.listdir(dir_name)
        dirs= sorted(dirs,key= Sorting_Dir)
        person1_dir = "person_{}".format(person1)
        person2_dir = "person_{}".format(person2)
        file_person1 = []
        file_person2 = []
        for i in range (0 ,5):
                row1=random.randint(1,30)
                row2=random.randint(1,30)

                file_name = "p{}_{}_L_{}.jpeg".format(person1,kind, row1)
                file_name1 = "p{}_{}_L_{}.jpeg".format(person2,kind, row2)
                file_to_search = path +'/'+ person1_dir + '/' + file_name
                file_to_search1 = path +'/'+ person2_dir + '/' + file_name1
            
                while(os.path.exists(file_to_search) == False):
                    print(file_to_search)
                    row1=random.randint(1,30)
                    file_name = "p{}_{}_L_{}.jpeg".format(person1,kind, row1)
                    file_to_search = path +'/'+ person1_dir + '/' + file_name
                while(os.path.exists(file_to_search1) == False):
                    row2=random.randint(1,30)
                    file_name1 = "p{}_{}_L_{}.jpeg".format(person2,kind, row2)
                    file_to_search1 = path +'/'+ person2_dir + '/' + file_name1
                file_person1.append(file_name)
                file_person2.append(file_name1)
        for img1 in file_person1:
            for img2 in file_person2:
                to_add = []
                to_add.append(person1_dir + '/' + img1)
                to_add.append(person2_dir + '/' + img2)
                to_add.append('1')
                diff_data.append(to_add)

        csv_file = pd.DataFrame(diff_data)
        csv_file = csv_file.sample(frac=1)
        csv_file = csv_file[0:8] 
        print('Done.')
        return csv_file

def from_two_pages_to_jpeg(data_path,folder="data1_as_one_page"):

    if not os.path.exists(folder):
        os.mkdir(folder)
        print('change the file type from 2 pages of type tiff to one jpeg image...')
        prepare_doc.prepare_doc_test(data_path,foleder=folder)
    print('Done.')

def creating_lines_for_each_person(path='data1_as_one_page',path_1="data_for_each_person"):

    if os.path.exists(path):

        print('Finding lines for each person...')

        files = os.listdir(path)

        if not os.path.exists(path_1):
            os.mkdir(path_1)
        for file in files:
            if file != '.DS_Store':
                number = int(file.split('-')[0])
                dir_name = 'person_{}'.format(number)
                if not os.path.exists(os.path.join(path_1, dir_name)):
                    os.mkdir(path_1 + '/' + dir_name)
                img = cv2.imread(path + '/' + file, 0)
                lines = detection_function.detect_lines(img)
                for indx, line in enumerate(lines):
                    to_save = Image.fromarray(line)
                    name = file.split('-')[1][0]
                    name = 'p' + str(number) + '_' + name + \
                        '_L_' + str((indx + 1))
                    to_save.save('{}/{}/{}.jpeg'.format(path_1,
                                                        dir_name, name))

        print('Done.')

def Delete_White_Lines(path='data_for_each_person'):
    # check if the path exist
    if os.path.exists(path):
        print('started deleting bad lines')
        dir_name = path
        # get the folder name
        dirs = os.listdir(dir_name)
        # get inside each person
        count = 0
        for _dir in dirs:
            if _dir != '.Ds_Store':
                files = os.listdir(dir_name+"/"+_dir)
                # get inside each img in person
                for indx, img in enumerate(files):
                    imgpath = dir_name+"/"+_dir+"/"+img
                    # read the image
                    img = cv2.imread(imgpath, 0)
                    # crop the left side of it
                    left_sideof_line = img[0:int(
                        img.shape[0]), 0:int(img.shape[1]/2)]
                    # crop the right side
                    right_sideof_line = img[0:int(img.shape[0]), int(
                        img.shape[1]/2):int(img.shape[1])]
                    # calculate the wihite px of ledt side
                    white_pixeles_for_left = np.count_nonzero(left_sideof_line)
                    # calculate the black px from the left side
                    Black_pixeles_for_left = left_sideof_line.size-white_pixeles_for_left
                    # calculate the wihite px of right side
                    white_pixeles_for_right = np.count_nonzero(
                        right_sideof_line)
                    # calculate the black px from the right side
                    Black_pixeles_for_right = right_sideof_line.size-white_pixeles_for_right
                    # if the wihte px in the right side equal to the size of the right side or smaler than it alittle
                    # so we will delete the line
                    if right_sideof_line.size - 1500 < white_pixeles_for_right <= right_sideof_line.size:
                        count += 1
                        os.remove(imgpath)
        print('Done, number of deleted lines: {}'.format(count))

def Sorting_Dir(dirname):
    return int(dirname.split("_")[1])

def find_match_pairs(path='data_for_each_person',start=0,end=39,filename='match_labels.csv'):

    if os.path.exists(path):
        print('Creating all possible pairs of lines that creat a Match and store the labels in csv file...')

        genuin_data = []

        dir_name = path
        dirs = os.listdir(dir_name)
        dirs= sorted(dirs,key= Sorting_Dir)

        for _dir in dirs[start:end]:
            if _dir != '.DS_Store':
                files = os.listdir(dir_name + '/' + _dir)
                for indx, img1 in enumerate(files):
                    for img2 in files[indx:]:
                        to_add = []
                        if img1 != img2:
                            to_add.append(_dir + '/' + img1)
                            to_add.append(_dir + '/' + img2)
                            to_add.append('0')
                            genuin_data.append(to_add)

        csv_file = pd.DataFrame(genuin_data)
        csv_file=csv_file.sample(frac=1)

        csv_file.to_csv(filename, index=False, sep=',', header=0)

        print('Done.')

def find_miss_match_pairs(path='data_for_each_person',start= 0,end= 39,filename='miss_match_labels.csv'):

    if os.path.exists(path):

        print('Creating all pairs of lines that creat a Miss Match and store the labels in csv file...')

        diff_data = []
        dir_name = path
        dirs = os.listdir(dir_name)
        dirs= sorted(dirs,key= Sorting_Dir)
        dirs=dirs[start:end]
        for indx, _dir in enumerate(dirs):
            if _dir != '.DS_Store':
                files_1 = os.listdir(dir_name + '/' + _dir)
                for _dir_ in dirs[indx:]:
                    if _dir_ != '.DS_Store':
                        files_2 = os.listdir(dir_name + '/' + _dir_)
                        if _dir != _dir_:
                            for img1 in files_1:
                                for img2 in files_2:
                                    to_add = []
                                    to_add.append(_dir + '/' + img1)
                                    to_add.append(_dir_ + '/' + img2)
                                    to_add.append('1')
                                    diff_data.append(to_add)

        csv_file = pd.DataFrame(diff_data)
        csv_file=csv_file.sample(frac=1)
        csv_file.to_csv(filename,
                        index=False, sep=',', header=0)

        print('Done.')

def create_label_file(file_1, file_2, num,filename):
    match_pairs = pd.read_csv(file_1, header= None, sep= ',',nrows= num )
    miss_match_pairs = pd.read_csv(file_2,header= None ,sep= ',',nrows= num )
    labels = pd.concat([match_pairs, miss_match_pairs])
    labels= labels.sample(frac=1)
    labels.to_csv(filename, index=False, sep=',', header=0)

def resize_image(dir_name="data_for_each_person"):
     if os.path.exists(dir_name):
        print('Creating all possible pairs of lines that creat a Match and store the labels in csv file...')
     dirs = os.listdir(dir_name)
     for _dir in dirs:
         if _dir !='.DS_Store':
             files = os.listdir(dir_name + '/' + _dir)
             for indx , img1 in enumerate(files):
                imgpath = dir_name+"/"+_dir+"/"+img1
                img = Image.open(imgpath)
                img = img.resize((1760,70))
                img.save(imgpath)

def rename_newData_files(path, start_number):
    files = os.listdir(path)
    files.sort(key=prepare_doc.bigger_than)
    flag=0
    for file in files:
        new_path= path+"/"+str(start_number)+"-"+file.split("-")[1]
        os.rename(path+"/"+file,new_path)
        flag+=1
        if flag == 2 : 
            flag=0
            start_number+=1


if __name__ == '__main__':
    print('Starting the preparing phase...')
    #from_two_pages_to_jpeg("C:/Users/97258/Desktop/wave2 data")
    #creating_lines_for_each_person()
    #Delete_White_Lines()
    # find_match_pairs(start=0,end=30,filename="Train_match_labels.csv")
    # find_match_pairs(start=30,end=39,filename="Test_match_labels.csv")
    # find_miss_match_pairs(start=0,end=30,filename="Train_miss_match_labels.csv")
    # find_miss_match_pairs(start=30,end=39,filename="Test_miss_match_labels.csv")
    # create_label_file('Train_match_labels.csv', 'Train_miss_match_labels.csv', 5000, "Train_Labels.csv")
    # create_label_file('Test_match_labels.csv', 'Test_miss_match_labels.csv', 2000, "Test_Labels.csv")
    # print('Done. Now you can use the data')
    # resize_image()
    # new_func('test_labels_try.csv')
