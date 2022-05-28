from email import header
from statistics import median
from unittest import result
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from cv2 import mean
from numpy import std
import dataManpiulation.detection_function as detection_function
import cv2
from PIL import Image
import os
import pandas as pd
import random
import dataManpiulation.prepare_data as prepare_data
# from resnet18 import ResNet18
from dataSets.data_set import LinesDataSet

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import dataManpiulation.findminmum as findminmum
from multiprocessing import Queue
from models.customResNet import ResNet, ResidualBlock

# from customResNet import ResNet, ResidualBlock
Proc_step=0

def update_excel(excel_file):
    test_file = pd.read_excel(excel_file)
    
    test_file["firstt"] = [ i.split('-')[0] for i in  test_file['first'] ]
    test_file["secondd"] = [ i.split('-')[0] for i in  test_file['second'] ]
    test_file['all'] = [ test_file['pfs_median'][indx] / (0.5*(test_file['pfirst_median'][indx]+test_file['psecond_median'][indx]))  for indx,_ in enumerate(test_file['pfirst_median'])]
    test_file['all2'] = [ test_file['pfs'][indx] / (0.5*(test_file['pfirst'][indx]+test_file['psecond'][indx]))  for indx,_ in enumerate(test_file['pfirst_median'])]
    arr =[]
    for indx , _ in enumerate(test_file['all']):
        if test_file['firstt'][indx] == test_file['secondd'][indx]:
            if test_file['all'][indx] >= 0.5:
                arr.append(1)
            else:
                arr.append(0)
        else:
            if test_file['all'][indx] > 0.5:
                arr.append(0)
            else:
                arr.append(1)
    arr1 =[]
    for indx , _ in enumerate(test_file['all2']):
        if test_file['firstt'][indx] == test_file['secondd'][indx]:
            if test_file['all2'][indx] >= 0.5:
                arr1.append(1)
            else:
                arr1.append(0)
        else:
            if test_file['all2'][indx] > 0.5:
                arr1.append(0)
            else:
                arr1.append(1)
    test_file['result'] = arr

    test_file['result2'] = arr1
    test_file.to_csv("final_GUI.csv")
    return "final_GUI.csv" ,test_file['result'].mean(),test_file['result2'].mean() 


def step_excel():
    return Proc_step    


def test ( model,test_loader, acc_history, train_flag, acc_history_std , acc_history_median):
    model.eval()
    count =0
    acc_nean = []
    acc_std =[]
    acc_median =[]
    for indx, data in enumerate(test_loader):
        image1, image2, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        label = label.cuda()
        with torch.no_grad():
            output = model(image1, image2)
            resultt= mean(output[:,0].cpu().numpy())[0]
            resultstd = std(output[:,0].cpu().numpy())
            result_median = median(output[:,0].cpu().numpy())
            acc_nean.append(resultt)
            acc_std.append(resultstd)
            acc_median.append(result_median)
            count+=1
        if count == 3:
            # 10 10 10 
            # 50  50  60 / 3 
            r = sum(acc_nean) /3  
            r_std =sum(acc_std)/3  
            r_median = sum(acc_median)/3    
            acc_history.append(r)
            acc_history_std.append(r_std)
            acc_history_median.append(r_median)
            count=0
            acc_nean = []
            acc_std =[]
            acc_median = []
    return acc_history , acc_history_std, acc_history_median


def creating_lines_for_each_file(path, path_1):
    """Create lines for each writer for any language split with . 
    params:  path str folder as one page
    params: path1  folder wanna fill it  witt lines for each person
    """
    if os.path.exists(path):
        print('Finding lines for each person...')
        files = os.listdir(path)
        if not os.path.exists(path_1):
            os.mkdir(path_1)
        for file in files:
            if file != '.DS_Store':
                number = file.split('.')[0]
                dir_name = 'person_{}'.format(number)
                if not os.path.exists(os.path.join(path_1, dir_name)):
                    os.mkdir(path_1 + '/' + dir_name)
                img = cv2.imread(path + '/' + file, 0)
                lines = detection_function.detect_lines(img)
                for indx, line in enumerate(lines):
                    to_save = Image.fromarray(line)
                    name = 'p'+str(number) + "_L_"+str((indx + 1))
                    to_save.save('{}/{}/{}.jpeg'.format(path_1,
                                                        dir_name, name))
        print('Done.')



def find_miss_match_pairs_two_writer(path='../data2_for_each_person',person1 = 1, person2 = 2 ):
    
    if os.path.exists(path):

        # print('Creating all pairs of lines that creat a Miss Match and store the labels in csv file...')
        diff_data = []
        dir_name = path
        dirs = os.listdir(dir_name)
        person1_dir = "person_{}".format(person1)
        person2_dir = "person_{}".format(person2)
        file_person1 = []
        file_person2 = []
        for i in range (0 ,12):
                row1=random.randint(1,30)
                row2=random.randint(1,30)
                file_name  = "p{}_L_{}.jpeg".format(person1,row1)
                file_name1 = "p{}_L_{}.jpeg".format(person2,row2)
                file_to_search  = path +'/'+ person1_dir + '/' + file_name
                file_to_search1 = path +'/'+ person2_dir + '/' + file_name1
                while(os.path.exists(file_to_search) == False):
                    row1=random.randint(1,30)
                    file_name  = "p{}_L_{}.jpeg".format(person1,row1)
                    file_to_search = path +'/'+ person1_dir + '/' + file_name
                while(os.path.exists(file_to_search1) == False):
                    row2=random.randint(1,30)
                    file_name1 = "p{}_L_{}.jpeg".format(person2,row2)
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
        csv_file.to_csv("filename1.csv",index=False, sep=',', header=0)
        csv_file = csv_file.sample(frac=1)
        csv_file = csv_file[0:30] 
        # print('Done.')
        return csv_file


def find_match_pairs_two_writer(path = "../data2_for_each_person" , person =1, flag= False):
    if os.path.exists(path):
        # print('Creating all possible pairs of lines that creat a Match and store the labels in csv file...')
        dir_name = path
        genuin_data = []
        dirs = os.listdir(dir_name)
        dirs_to_search= "person_{}".format(person)
        for _dir in dirs:
            if _dir != '.DS_Store':
                files = os.listdir(dir_name + '/' + _dir)
                file =[]
                if _dir == dirs_to_search:
                    linee_number = []
                    for i in range(0,12):
                        row = random.randint(1,30)
                        file_name = "p{}_L_{}.jpeg".format(person, row)
                        file_to_search = dir_name + '/' + _dir + '/' + file_name
                        while(os.path.exists(file_to_search) == False) and int(row) not in linee_number:
                            row=random.randint(1,30)
                            file_name = "p{}_L_{}.jpeg".format(person, row)
                            file_to_search= dir_name + '/' + _dir + '/' + file_name
                        file.append(file_name)
                        linee_number.append(row)
                    for indx, img1 in enumerate(file):
                        for img2 in file[indx:]:
                            to_add = []
                            if img1 != img2:
                                to_add.append(_dir + '/' + img1)
                                to_add.append(_dir + '/' + img2)
                                to_add.append('0')
                                genuin_data.append(to_add)
        csv_file = pd.DataFrame(genuin_data)
        csv_file = csv_file.sample(frac=1)
        if flag == 1:
            csv_file = csv_file[0:31]
        else:
            csv_file = csv_file[0:30]




        return csv_file


def detect_lines(Excel_file,datapath):
    prepare_data.from_two_pages_to_jpeg(datapath,"Motaz_as_one_page")
    creating_lines_for_each_file("Motaz_as_one_page","Motaz_for_each_Person")
    prepare_data.Delete_White_Lines("Motaz_for_each_Person")
    min = 1
    count =0
    while min < 55:
       min = findminmum.find_min("Motaz_for_each_Person")
       count+=1
    print(count)   
    prepare_data.resize_image("Motaz_for_each_Person")


def testing(filename1 ,model_path):

    acc_history = []
    acc_history_std =[]
    acc_history_median =[]
    test_data_set = LinesDataSet(filename1, '../data2_for_each_person', transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))  ]))
    test_line_data_loader = DataLoader(test_data_set, shuffle=False, batch_size=10)
    model = ResNet(ResidualBlock, [2, 2, 2])

    model = model.cuda()
    model.load_state_dict(torch.load(model_path, map_location = 'cuda:0'))
    test(model, test_line_data_loader, acc_history=acc_history, acc_history_std= acc_history_std, train_flag=False, acc_history_median= acc_history_median)
    return acc_history , acc_history_std, acc_history_median


def looping_into_excel(Excel_file,model_path,que,que2):
    excel_file=[]
    resultss= []
    csv_file = pd.DataFrame(excel_file)
    max_rows = Excel_file.shape[0]
    for i in range(max_rows):
        que.put(i+1)
        excel_file=[]
        toadd=[]
        #create data frame 
        csv_file = pd.DataFrame(excel_file)
        #take the first cell extract the name of the file
        first_file=Excel_file.loc[i][0]
        #take the second cell extract the name of the file
        second_file=Excel_file.loc[i][1]
        # print(first_file + "," + second_file)
        global Proc_step
        Proc_step = (i+1)/max_rows       
        first_file_number = first_file.split(".")[0]
        second_file_number = second_file.split(".")[0]
        #name file excel
        filename1 = "match_Label.csv"
        #find match pairs for the first person
        first_csv = find_match_pairs_two_writer(person = first_file_number,flag=True)
        #find match pairs for the second person
        second_csv = find_match_pairs_two_writer(person = second_file_number)
        #concat the result to excel file
        csv_file = pd.concat([csv_file,first_csv])
        third_csv =  find_miss_match_pairs_two_writer( person1=first_file_number, person2=second_file_number)
        csv_file = pd.concat([csv_file,third_csv])
        csv_file = pd.concat([csv_file,second_csv])
        csv_file.to_csv(filename1,index=False, sep=',', header=0)
        # print("Start testing")
        results , result_std , result_median= testing(filename1=filename1,model_path=model_path)
        toadd.append(first_file)
        toadd.append(second_file)

        toadd.append(results[0])
        toadd.append(result_std[0])
        toadd.append(result_median[0])

        toadd.append(results[1])
        toadd.append(result_std[1])
        toadd.append(result_median[1])

        toadd.append(results[2])
        toadd.append(result_std[2])
        toadd.append(result_median[2])
        que2.put(toadd)
        resultss.append(toadd)
    main_Excel = pd.DataFrame(resultss)
    headerr = ["first","second","pfirst","pfirst_std","pfirst_median","pfs","pfs_std","pfs_median","psecond","psecond_std","psecond_median" ]
    main_Excel.to_excel("final1.xlsx",index=False,header=headerr)
    return "final1.xlsx"


def testing_excel(excel_path, data_path,que,que2):
    print("Starting reading excel file")
    test_file = pd.read_excel(excel_path)
    detect_lines(test_file,data_path)
    excel_fil,median_avg, mean_avg=main_test(test_file=test_file,model_path=r"C:\Users\97258\Desktop\custom_resnet\custom_ResNet_without_reg_writers_on_arabic_with_weight_decay_pre_trained\model_0_epoch_30.pt",que=que,que2=que2)
    return excel_fil


def main_test(test_file, model_path,que,que2):
    excel = looping_into_excel(test_file,model_path=model_path,que=que,que2=que2)
    excel_file,median_avg, mean_avg = update_excel(excel)
    print(median_avg, mean_avg)
    return median_avg, mean_avg


def creating_excel_for_testing_3(excel_file1,excel_file2):
    test_file = pd.read_excel(excel_file1,skiprows=1,header=None)
    print(test_file)
    file1 = pd.read_excel(excel_file2,skiprows=1,header=None)
    print(file1)
    max_row = file1.shape[0]
    excel_ = []
    num = []
    for i in range(test_file.shape[0]):
        toadd=[]
        randoom = random.randint(1,max_row-1)
        while randoom in num:
            randoom = random.randint(1,max_row-1)
        num.append(randoom)
        leftt= file1.iloc[randoom][0]
        rightt= file1.iloc[randoom][1]
        toadd.append(leftt)
        toadd.append(rightt)
        excel_.append(toadd)
    exceell= pd.DataFrame(excel_)
    headerr = ["first","second"]

    main_excel=pd.concat([test_file,exceell],ignore_index=False)
    main_excel.to_excel("../testing3.xlsx",index=False, header=headerr)


def creating_excel_for_testing_2(excel_file):
    test_file = pd.read_excel(excel_file,header=None,sheet_name="70D-test")
    max_row = test_file.shape[0]
    excel_ = []
    for i in range(0,max_row,1):
        toadd=[]
        left_writer = test_file.iloc[i][0]
        for j in range(0,max_row,1):
            if i != j:
                toadd=[]
                right_writer = test_file.iloc[j][0]
                toadd.append(left_writer)
                toadd.append(right_writer)
                excel_.append(toadd)
    main_excel = pd.DataFrame(excel_)
    headerr = ["first","second"]
    main_excel.to_excel("../testing2.xlsx",index=False,header=headerr)


def create_excel_for_testing(excel_file):
    test_file = pd.read_excel(excel_file,header=None,sheet_name="70D-test")
    max_row = test_file.shape[0]
    excel_ =[]
    for i in range(0,max_row,1):
        toadd=[]
        left_writer =  test_file.iloc[i][0]
        toadd.append(left_writer)
        toadd.append(left_writer)
        excel_.append(toadd)
    main_excel = pd.DataFrame(excel_)
    headerr = ["first","second"]
    main_excel.to_excel("../testing.xlsx",index=False,header=headerr)
    return 


def create_excel_for_testing4(excel_file):
    test_file = pd.read_excel(excel_file,header=None,sheet_name="70D-test")
    max_row = test_file.shape[0]
    excel_ =[]
    for i in range(0,max_row,2):
        toadd = []
        left_writer =  test_file.iloc[i][0]
        number = left_writer.split("-")[0]
        writer = number+"-b.tiff"
        toadd.append(left_writer)
        toadd.append(writer)
        excel_.append(toadd)
    main_excel = pd.DataFrame(excel_)
    headerr = ["first","second"]
    main_excel.to_excel("../testing4.xlsx",index=False,header=headerr)

def median_mean_test():
    # test_file = testing_excel(r"testing3.xlsx",r"C:\Users\97258\Desktop\Motaz")
    que1 = Queue()
    que2 = Queue()
    test_file = pd.read_excel("../testing3.xlsx")
    result1 =[]
    for i in range(0,30):
        print("epoch",i+1)
        to_add =[]
        model_path = r'../model_0_epoch_{}.pt'.format(i+1)
        median_avg, mean_avg= main_test(test_file=test_file,model_path=model_path,que= que1, que2= que2)
        to_add.append(i+1)
        to_add.append(median_avg)
        to_add.append(mean_avg)
        result1.append(to_add)
    excell= pd.DataFrame(result1)
    headerr = ['step','median','mean']
    excell.to_excel('hebrew_median.xlsx',index=False,header=headerr)

# create_excel_for_testing(r"C:\Users\FinalProject\Desktop\Motaz_test.xlsx")
# creating_excel_for_testing_2(r"C:\Users\FinalProject\Desktop\Motaz_test.xlsx")
# creating_excel_for_testing_3("../testing.xlsx","../testing2.xlsx")
# create_excel_for_testing4(r"C:\Users\FinalProject\Desktop\Motaz_test.xlsx")
