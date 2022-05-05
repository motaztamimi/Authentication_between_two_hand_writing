import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from  dataManpiulation.prepare_data import from_two_pages_to_jpeg, Delete_White_Lines, resize_image
from  dataManpiulation.detection_function import detect_lines
from dataManpiulation.findminmum import find_min
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import random
import cv2
import torch.nn.functional as F

from PIL import Image
from dataSets.data_set import LinesDataSetTripletWithLabel
from models.triplet_model import ResidualBlock , ResNet

def update_excel(excel_file):
    test_file = pd.read_excel(excel_file)
    
    test_file["firstt"] = [ i.split('-')[0] for i in  test_file['first'] ]
    test_file["secondd"] = [ i.split('-')[0] for i in  test_file['second'] ]
    # test_file['all'] = [ test_file['pfs_median'][indx] / (0.5*(test_file['pfirst_median'][indx]+test_file['psecond_median'][indx]))  for indx,_ in enumerate(test_file['pfirst_median'])]
    test_file['all2'] = [ test_file['pfs'][indx] / (0.5*(test_file['pfirst'][indx]+test_file['psecond'][indx]))  for indx,_ in enumerate(test_file['pfirst'])]
    # arr =[]
    # for indx , _ in enumerate(test_file['all']):
    #     if test_file['firstt'][indx] == test_file['secondd'][indx]:
    #         if test_file['all'][indx] >= 0.5:
    #             arr.append(1)
    #         else:
    #             arr.append(0)
    #     else:
    #         if test_file['all'][indx] > 0.5:
    #             arr.append(0)
    #         else:
    #             arr.append(1)
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
    # test_file['result'] = arr

    test_file['result2'] = arr1
    test_file.to_csv("final_GUI.csv")
    return "final_GUI.csv" ,test_file['result2'].mean() 

def test(model, test_loader, acc_history, thresh):
    model.eval()
    acc  = []
    for _, data in enumerate(test_loader):
        image1, image2, image3, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image3 = image3.float().cuda()
        image1 = image1[:, None, :, :]
        image2 = image2[:, None, :, :]
        image3 = image3[:, None, :, :]
        label = label.cuda()
        with torch.no_grad():
            output = model(image1, image2, image3)
            dist_a_p = F.pairwise_distance(output[0], output[1], 2)
            dist_a_n = F.pairwise_distance(output[0], output[2], 2)
            pred = (dist_a_n - dist_a_p - thresh).cpu().data
            pred = pred.reshape(pred.size()[0], 1)
            acc.append(( ( (pred <= 0).sum())/ dist_a_p.size()[0]).data.item())
            
            acc_history.append(acc[0])
            acc = []  
    return acc_history


def testing(filename1 ,model_path):
    
    acc_history = []
    acc_history_miss = []
    acc_history_std =[]
    acc_history_median =[]
    model = ResNet(ResidualBlock, [2, 2, 2]).cuda()
    test_data_set = LinesDataSetTripletWithLabel(filename1, '../Motaz_for_each_Person', transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]))
    test_line_data_loader = DataLoader(test_data_set, shuffle=False, batch_size=30)
    torch.manual_seed(17)
    model.load_state_dict(torch.load(model_path, map_location = 'cuda:0'))
    test(model, test_line_data_loader, acc_history=acc_history, thresh =2)
    print(acc_history)
    print(acc_history_miss)
    
    return acc_history , acc_history_std, acc_history_median



def find_match_pairs_for_two_writer_triplet(path = "../Motaz_for_each_Person",person = -1,person2 = -1,miss_match_flag = False, person1_ancor = -1,Label = 0,full_batch= False) :
    """this function find match pairs and miss match pairs for 2 or 1 person  
    miss_match_flag : True if we want to finnd miss match between to writer 
    """
    if os.path.exists(path):
        dir_name = path
        genuin_data = []
        dirs = os.listdir(dir_name)
        negattive_path = ""
        dir_to_search = "person_{}".format(person)
        dir_to_search_person2 =""
        if miss_match_flag:
            dir_to_search_person2 = "person_{}".format(person2)
        if miss_match_flag:
            Ancor_row = person1_ancor 
        else:
            Ancor_row = random.randint(1,30)
        Ancor_file_name = "p{}_L_{}.jpeg".format(person, Ancor_row)
        Ancor_file_to_search = dir_name + '/' + dir_to_search + '/' + Ancor_file_name
        while(os.path.exists(Ancor_file_to_search) == False):
            Ancor_row = random.randint(1,30)
            Ancor_file_name = "p{}_L_{}.jpeg".format(person, Ancor_row)
            Ancor_file_to_search = dir_name + '/' + dir_to_search + '/' + Ancor_file_name 
        Ancor_path =  dir_to_search + '/' + Ancor_file_name                 
        for  i in range(30):
            to_add = []
            positive_row = random.randint(1,35)
            positive_file_name = "p{}_L_{}.jpeg".format(person, positive_row)
            positive_file_to_search = dir_name + '/' + dir_to_search + '/' +positive_file_name                    
            while ( os.path.exists(positive_file_to_search) == False or  positive_row == Ancor_row):
                positive_row = random.randint(1,35)
                positive_file_name = "p{}_L_{}.jpeg".format(person, positive_row)
                positive_file_to_search = dir_name + '/' + dir_to_search + '/' +positive_file_name  
            positive_path = dir_to_search + '/' + positive_file_name
            if miss_match_flag:                        
                negative_row = random.randint(1,35)
                negative_file_name = "p{}_L_{}.jpeg".format(person2, negative_row)
                negative_file_to_search = dir_name + '/' + dir_to_search_person2 + '/' +negative_file_name                    
                while ( os.path.exists(negative_file_to_search) == False  ) :
                    negative_row = random.randint(1,35)
                    negative_file_name = "p{}_L_{}.jpeg".format(person2, negative_row)
                    negative_file_to_search = dir_name + '/' + dir_to_search_person2 + '/' +negative_file_name
                negattive_path = dir_to_search_person2 + '/' +negative_file_name
            else:
                negative_row = random.randint(1,35)
                negative_file_name = "p{}_L_{}.jpeg".format(person, negative_row)
                negative_file_to_search = dir_name + '/' + dir_to_search + '/' +negative_file_name                    
                while ( os.path.exists(negative_file_to_search) == False  or  negative_row == Ancor_row  or  negative_row == positive_row ) :
                    negative_row = random.randint(1,35)
                    negative_file_name = "p{}_L_{}.jpeg".format(person, negative_row)
                    negative_file_to_search = dir_name + '/' + dir_to_search + '/' +negative_file_name
                negattive_path = dir_to_search + '/' +negative_file_name
            
            to_add.append(Ancor_path)
            to_add.append(positive_path)
            to_add.append(negattive_path)
            to_add.append(Label)
            genuin_data.append(to_add)
            if full_batch:
                to_add = []
                to_add.append(Ancor_path)
                to_add.append(positive_path)
                to_add.append(negattive_path)
                to_add.append(Label)
                full_batch= False
                genuin_data.append(to_add)
        csv_file = pd.DataFrame(genuin_data)
        csv_file = csv_file.sample(frac=1)
        if miss_match_flag:
            return csv_file
        return csv_file, Ancor_row



def looping_into_excel(Excel_file,model_path):
    excel_file=[]
    resultss= []
    csv_file = pd.DataFrame(excel_file)
    max_rows=Excel_file.shape[0]
    for i in range(max_rows):
        # que.put(i+1)
        excel_file=[]
        toadd=[]
        #create data frame 
        csv_file = pd.DataFrame(excel_file)
        #take the first cell extract the name of the file
        first_file=Excel_file.loc[i][0]
        #take the second cell extract the name of the file
        second_file=Excel_file.loc[i][1]
        # print(first_file + "," + second_file)     
        first_file_number = first_file.split(".")[0]
        second_file_number = second_file.split(".")[0]
        print(first_file , ", " , second_file)
        #name file excel
        filename1 = "../match_Label.csv"
        #find match pairs for the first person
        first_csv, person1_ancor = find_match_pairs_for_two_writer_triplet(person = first_file_number,full_batch = True)
        #find match pairs for the second person
        second_csv, person2_ancor  = find_match_pairs_for_two_writer_triplet(person = second_file_number)
        #concat the result to excel file
        csv_file = pd.concat([csv_file,first_csv])
        if first_file_number == second_file_number:
            third_csv =  find_match_pairs_for_two_writer_triplet(person=first_file_number, person2=second_file_number,miss_match_flag=True,person1_ancor= person1_ancor,Label = 0)
        else:
            third_csv =  find_match_pairs_for_two_writer_triplet(person=first_file_number, person2=second_file_number,miss_match_flag=True,person1_ancor= person1_ancor,Label = 1)
        csv_file = pd.concat([csv_file,third_csv])
        csv_file = pd.concat([csv_file,second_csv])
        csv_file.to_csv(filename1,index=False, sep=',', header=0)
        print("Start testing")
      
        results , result_std , result_median= testing(filename1=filename1,model_path=model_path)
        toadd.append(first_file)
        toadd.append(second_file)

        toadd.append(results[0])
        # toadd.append(result_std[0])
        # toadd.append(result_median[0])

        toadd.append(results[1])
        # toadd.append(result_std[1])
        # toadd.append(result_median[1])

        toadd.append(results[2])
        # toadd.append(result_std[2])
        # toadd.append(result_median[2])
        # que2.put(toadd)
        resultss.append(toadd)
    main_Excel = pd.DataFrame(resultss)
    headerr= ["first","second","pfirst","pfs","psecond"]
    main_Excel.to_excel("final1.xlsx",index=False,header=headerr)
    return "final1.xlsx"

def creating_lines_for_each_file(path='../data1_as_one_page',path_1="../data_for_each_person"):
    
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
                lines =  detect_lines(img)
                for indx, line in enumerate(lines):
                    to_save = Image.fromarray(line)
                    name = file.split('-')[1][0]
                    name = 'p'+str(number) + "_L_"+str((indx + 1))
                    to_save.save('{}/{}/{}.jpeg'.format(path_1,
                                                        dir_name, name))
        print('Done.')


def detect_liness(Excel_file,datapath):
    from_two_pages_to_jpeg(datapath,"../Motaz_as_one_page")
    creating_lines_for_each_file("../Motaz_as_one_page","../Motaz_for_each_Person")
    Delete_White_Lines("../Motaz_for_each_Person")
    min = 1
    count =0
    while min < 55:
       min = find_min("../Motaz_for_each_Person")
       count+=1
    print(count)   
    resize_image("../Motaz_for_each_Person")

def main_test(test_file, model_path):
    excel =looping_into_excel(test_file,model_path=model_path)
    excel_file, mean_avg = update_excel(excel)
    # print( median_avg, mean_avg)
    # return excel_file,median_avg, mean_avg
    pass

def testing_excel(excel_path, data_path):
    print("Starting reading excel file")
    test_file = pd.read_excel(excel_path)
    # detect_liness(test_file,data_path)
    main_test(test_file=test_file,model_path=r"C:\Users\97258\Desktop\model_epoch_20.pt")
    # return excel_fil 


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
    test_file = pd.read_excel(excel_file,header=None)
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
    test_file = pd.read_excel(excel_file,header=None,)
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


if __name__ == "__main__":
    testing_excel("../testing3.xlsx", r"C:\Users\97258\Desktop\Motaz")
    # excel_file, mean_avg = update_excel("final1.xlsx")
    # create_excel_for_testing(r"C:\Users\97258\Desktop\Motaz_test.xlsx")
    # creating_excel_for_testing_2(r"C:\Users\97258\Desktop\Motaz_test.xlsx")
    # creating_excel_for_testing_3("../testing.xlsx","../testing2.xlsx")    