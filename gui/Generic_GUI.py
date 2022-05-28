import torch.nn.functional as F
from statistics import median
from cv2 import mean
from dataSets.data_set import LinesDataSetTripletWithLabel
from numpy import std
import dataManpiulation.detection_function as detection_function
import cv2
from PIL import Image
import pandas as pd
import random
import dataManpiulation.prepare_data as prepare_data
from dataSets.data_set import LinesDataSet
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import dataManpiulation.findminmum as findminmum
from multiprocessing import Queue
from models.customResNet import ResNet, ResidualBlock
import shutil
import os
import sys
from sphinx import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def testing_excel(excel_path, data_path, model, que, que2, mode=False):
    print("Starting reading excel file")
    test_file = pd.read_excel(excel_path)
    # folder, GUI_for_each_person = detect_lines(data_path)
    excel = main_test(test_file=test_file, model_path=model,
                      data_for_each_person="../GUI_data_for_each_person", que=que, que2=que2, mode=mode)
    return excel


def detect_lines(datapath):
    folder = "../GUI_data_as_one_page"
    GUI_for_each_person = "../GUI_data_for_each_person"
    prepare_data.from_two_pages_to_jpeg(datapath, folder=folder)
    creating_lines_for_each_file(folder, GUI_for_each_person)
    prepare_data.Delete_White_Lines(GUI_for_each_person)
    min = 1
    count = 0
    while min < 55:
        min = findminmum.find_min(GUI_for_each_person)
        count += 1
    print(count)
    prepare_data.resize_image(GUI_for_each_person)
    return folder, GUI_for_each_person


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


def main_test(test_file, model_path, data_for_each_person, que, que2, mode):
    if mode:
        excel = looping_into_excel_triplet_mode(
            test_file, model_path=model_path, data_for_each_person=data_for_each_person, que=que, que2=que2, mode=mode)
    else:
        excel = looping_into_excel(test_file, model_path=model_path,
                                   data_for_each_person=data_for_each_person, que=que, que2=que2, mode=mode)
    # excel_file ,median_avg, mean_avg = update_excel(excel)
    return excel
    # print(median_avg, mean_avg)
    # return median_avg, mean_avg


def looping_into_excel(Excel_file, model_path, data_for_each_person, que, que2, mode):
    excel_file = []
    resultss = []
    csv_file = pd.DataFrame(excel_file)
    max_rows = Excel_file.shape[0]
    for i in range(max_rows):
        que.put(i+1)

        excel_file = []
        toadd = []
        # create data frame
        csv_file = pd.DataFrame(excel_file)
        # take the first cell extract the name of the file
        first_file = Excel_file.loc[i][0]
        # take the second cell extract the name of the file
        second_file = Excel_file.loc[i][1]
        # print(first_file + "," + second_file)
        global Proc_step
        Proc_step = (i+1)/max_rows
        first_file_number = first_file.split(".")[0]
        second_file_number = second_file.split(".")[0]
        # name file excel
        filename1 = "../match_Label.csv"
        # find match pairs for the first person
        first_csv = find_match_pairs_two_writer_generic(
            path=data_for_each_person, person=first_file_number, flag=True)
        # find match pairs for the second person
        second_csv = find_match_pairs_two_writer_generic(
            path=data_for_each_person, person=second_file_number)
        # concat the result to excel file
        csv_file = pd.concat([csv_file, first_csv])
        third_csv = find_miss_match_pairs_two_writer_generic(
            path=data_for_each_person, person1=first_file_number, person2=second_file_number)
        csv_file = pd.concat([csv_file, third_csv])
        csv_file = pd.concat([csv_file, second_csv])
        csv_file.to_csv(filename1, index=False, sep=',', header=0)
        print("Start testing")
        results, result_std, result_median = testing(
            filename1=filename1, model_path=model_path, data_for_each_person=data_for_each_person, mode=mode)
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
    headerr = ["first", "second", "pfirst", "pfirst_std", "pfirst_median",
               "pfs", "pfs_std", "pfs_median", "psecond", "psecond_std", "psecond_median"]
    main_Excel.to_excel("../final1.xlsx", index=False, header=headerr)
    # TODO must return filename to
    # TODO  delete the header
    # delete it
    return "../final1.xlsx"


def looping_into_excel_triplet_mode(Excel_file, model_path, data_for_each_person, que, que2, mode):
    excel_file = []
    resultss = []
    csv_file = pd.DataFrame(excel_file)
    max_rows = Excel_file.shape[0]
    for i in range(max_rows):
        que.put(i+1)
        excel_file = []
        toadd = []
        # create data frame
        csv_file = pd.DataFrame(excel_file)
        # take the first cell extract the name of the file
        first_file = Excel_file.loc[i][0]
        # take the second cell extract the name of the file
        second_file = Excel_file.loc[i][1]
        # print(first_file + "," + second_file)
        first_file_number = first_file.split(".")[0]
        second_file_number = second_file.split(".")[0]
        # print(first_file , ", " , second_file)
        # name file excel
        filename1 = "../match_Label.csv"
        # find match pairs for the first person
        first_csv, person1_ancor = find_miss_match_pairs_two_writer_generic_triplet(
            path=data_for_each_person, person=first_file_number, full_batch=True)
        # find match pairs for the second person
        second_csv, person2_ancor = find_miss_match_pairs_two_writer_generic_triplet(
            path=data_for_each_person, person=second_file_number)
        # concat the result to excel file
        csv_file = pd.concat([csv_file, first_csv])
        if first_file_number == second_file_number:
            third_csv = find_miss_match_pairs_two_writer_generic_triplet(
                path=data_for_each_person, person=first_file_number, person2=second_file_number, miss_match_flag=True, person1_ancor=person1_ancor, Label=0)
        else:
            third_csv = find_miss_match_pairs_two_writer_generic_triplet(
                apth=data_for_each_person, person=first_file_number, person2=second_file_number, miss_match_flag=True, person1_ancor=person1_ancor, Label=1)
        csv_file = pd.concat([csv_file, third_csv])
        csv_file = pd.concat([csv_file, second_csv])
        csv_file.to_csv(filename1, index=False, sep=',', header=0)

        results, result_std, result_median = testing(
            filename1=filename1, model_path=model_path, mode=mode)
        toadd.append(first_file)
        toadd.append(second_file)

        toadd.append(results[0])

        toadd.append(results[1])
        toadd.append(results[2])
        que2.put(toadd)
        resultss.append(toadd)
    main_Excel = pd.DataFrame(resultss)
    headerr = ["first", "second", "pfirst", "pfs", "psecond"]
    main_Excel.to_excel("../final1.xlsx", index=False, header=headerr)
    return "../final1.xlsx"


def find_miss_match_pairs_two_writer_generic_triplet(path, person=-1, person2=-1, miss_match_flag=False, person1_ancor=-1, Label=0, full_batch=False):
    """this function find match pairs and miss match pairs for 2 or 1 person  
    miss_match_flag : True if we want to finnd miss match between to writer 
    """
    if os.path.exists(path):
        dir_name = path
        genuin_data = []
        dirs = os.listdir(dir_name)
        negattive_path = ""
        dir_to_search = "person_{}".format(person)
        dir_to_search_person2 = ""
        if miss_match_flag:
            dir_to_search_person2 = "person_{}".format(person2)
        if miss_match_flag:
            Ancor_row = person1_ancor
        else:
            Ancor_row = random.randint(1, 30)
        Ancor_file_name = "p{}_L_{}.jpeg".format(person, Ancor_row)
        Ancor_file_to_search = dir_name + '/' + dir_to_search + '/' + Ancor_file_name
        while(os.path.exists(Ancor_file_to_search) == False):
            Ancor_row = random.randint(1, 30)
            Ancor_file_name = "p{}_L_{}.jpeg".format(person, Ancor_row)
            Ancor_file_to_search = dir_name + '/' + dir_to_search + '/' + Ancor_file_name
        Ancor_path = dir_to_search + '/' + Ancor_file_name
        for i in range(30):
            to_add = []
            positive_row = random.randint(1, 35)
            positive_file_name = "p{}_L_{}.jpeg".format(person, positive_row)
            positive_file_to_search = dir_name + '/' + \
                dir_to_search + '/' + positive_file_name
            while (os.path.exists(positive_file_to_search) == False or positive_row == Ancor_row):
                positive_row = random.randint(1, 35)
                positive_file_name = "p{}_L_{}.jpeg".format(
                    person, positive_row)
                positive_file_to_search = dir_name + '/' + \
                    dir_to_search + '/' + positive_file_name
            positive_path = dir_to_search + '/' + positive_file_name
            if miss_match_flag:
                negative_row = random.randint(1, 35)
                negative_file_name = "p{}_L_{}.jpeg".format(
                    person2, negative_row)
                negative_file_to_search = dir_name + '/' + \
                    dir_to_search_person2 + '/' + negative_file_name
                while (os.path.exists(negative_file_to_search) == False):
                    negative_row = random.randint(1, 35)
                    negative_file_name = "p{}_L_{}.jpeg".format(
                        person2, negative_row)
                    negative_file_to_search = dir_name + '/' + \
                        dir_to_search_person2 + '/' + negative_file_name
                negattive_path = dir_to_search_person2 + '/' + negative_file_name
            else:
                negative_row = random.randint(1, 35)
                negative_file_name = "p{}_L_{}.jpeg".format(
                    person, negative_row)
                negative_file_to_search = dir_name + '/' + \
                    dir_to_search + '/' + negative_file_name
                while (os.path.exists(negative_file_to_search) == False or negative_row == Ancor_row or negative_row == positive_row):
                    negative_row = random.randint(1, 35)
                    negative_file_name = "p{}_L_{}.jpeg".format(
                        person, negative_row)
                    negative_file_to_search = dir_name + '/' + \
                        dir_to_search + '/' + negative_file_name
                negattive_path = dir_to_search + '/' + negative_file_name

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
                full_batch = False
                genuin_data.append(to_add)
        csv_file = pd.DataFrame(genuin_data)
        csv_file = csv_file.sample(frac=1)
        if miss_match_flag:
            return csv_file
        return csv_file, Ancor_row


def find_miss_match_pairs_two_writer_generic(path, person1=1, person2=2):

    if os.path.exists(path):

        # print('Creating all pairs of lines that creat a Miss Match and store the labels in csv file...')
        diff_data = []
        person1_dir = "person_{}".format(person1)
        person2_dir = "person_{}".format(person2)
        file_person1 = []
        file_person2 = []
        for i in range(0, 12):
            row1 = random.randint(1, 30)
            row2 = random.randint(1, 30)
            file_name = "p{}_L_{}.jpeg".format(person1, row1)
            file_name1 = "p{}_L_{}.jpeg".format(person2, row2)
            file_to_search = path + '/' + person1_dir + '/' + file_name
            file_to_search1 = path + '/' + person2_dir + '/' + file_name1
            while(os.path.exists(file_to_search) == False):
                row1 = random.randint(1, 30)
                file_name = "p{}_L_{}.jpeg".format(person1, row1)
                file_to_search = path + '/' + person1_dir + '/' + file_name
            while(os.path.exists(file_to_search1) == False):
                row2 = random.randint(1, 30)
                file_name1 = "p{}_L_{}.jpeg".format(person2, row2)
                file_to_search1 = path + '/' + person2_dir + '/' + file_name1
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
        csv_file.to_csv("../filename1.csv", index=False, sep=',', header=0)
        csv_file = csv_file.sample(frac=1)
        csv_file = csv_file[0:30]
        # print('Done.')
        return csv_file


def find_match_pairs_two_writer_generic(path, person, flag=False):
    if os.path.exists(path):
        dir_name = path
        genuin_data = []
        dirs_to_search = f"person_{person}"
        linee_number = []
        file = []
        for i in range(0, 12):
            row = random.randint(1, 30)
            file_name = "p{}_L_{}.jpeg".format(person, row)
            file_to_search = dir_name + '/' + dirs_to_search + '/' + file_name
            while(os.path.exists(file_to_search) == False) and int(row) not in linee_number:
                row = random.randint(1, 30)
                file_name = "p{}_L_{}.jpeg".format(person, row)
                file_to_search = dir_name + '/' + dirs_to_search + '/' + file_name
            file.append(file_name)
            linee_number.append(row)
        for indx, img1 in enumerate(file):
            for img2 in file[indx:]:
                to_add = []
                if img1 != img2:
                    to_add.append(dirs_to_search + '/' + img1)
                    to_add.append(dirs_to_search + '/' + img2)
                    to_add.append('0')
                    genuin_data.append(to_add)
        csv_file = pd.DataFrame(genuin_data)
        csv_file = csv_file.sample(frac=1)
        if flag == 1:
            csv_file = csv_file[0:31]
        else:
            csv_file = csv_file[0:30]
        return csv_file


def testing(filename1, model_path, data_for_each_person, mode):

    acc_history = []
    acc_history_std = []
    acc_history_median = []
    test_data_set = ""
    if mode:
        test_data_set = LinesDataSetTripletWithLabel(filename1, data_for_each_person, transform=transforms.Compose([
                                                     transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        test_line_data_loader = DataLoader(
            test_data_set, shuffle=False, batch_size=30)
    else:
        test_data_set = LinesDataSet(filename1, data_for_each_person, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        test_line_data_loader = DataLoader(
            test_data_set, shuffle=False, batch_size=10)
    model = ResNet(ResidualBlock, [2, 2, 2])
    model = model.cuda()
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    if mode:
        test_triplet(model, test_line_data_loader, acc_history, 1.5)
    else:
        test_cross(model, test_line_data_loader, acc_history=acc_history,
                   acc_history_std=acc_history_std, train_flag=False, acc_history_median=acc_history_median)
    return acc_history, acc_history_std, acc_history_median


def test_cross(model, test_loader, acc_history, train_flag, acc_history_std, acc_history_median):
    model.eval()
    count = 0
    acc_nean = []
    acc_std = []
    acc_median = []
    for indx, data in enumerate(test_loader):
        image1, image2, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image1 = image1[:, None, :, :]
        image2 = image2[:, None, :, :]
        label = label.cuda()
        with torch.no_grad():
            output = model(image1, image2)
            resultt = mean(output[:, 0].cpu().numpy())[0]
            resultstd = std(output[:, 0].cpu().numpy())
            result_median = median(output[:, 0].cpu().numpy())
            acc_nean.append(resultt)
            acc_std.append(resultstd)
            acc_median.append(result_median)
            count += 1
        if count == 3:
            r = sum(acc_nean) / 3
            r_std = sum(acc_std)/3
            r_median = sum(acc_median)/3
            acc_history.append(r)
            acc_history_std.append(r_std)
            acc_history_median.append(r_median)
            count = 0
            acc_nean = []
            acc_std = []
            acc_median = []
    return acc_history, acc_history_std, acc_history_median


def test_triplet(model, test_loader, acc_history, thresh):
    model.eval()
    acc = []
    # print(thresh)
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
            acc.append((((pred <= 0).sum()) / dist_a_p.size()[0]).data.item())

            acc_history.append(acc[0])
            acc = []
    return acc_history


def update_excel(excel_file):
    test_file = pd.read_excel(excel_file)
    test_file["firstt"] = [i.split('-')[0] for i in test_file['first']]
    test_file["secondd"] = [i.split('-')[0] for i in test_file['second']]
    test_file['all'] = [test_file['pfs_median'][indx] /
                        (0.5*(test_file['pfirst_median'][indx]+test_file['psecond_median'][indx])) for indx, _ in enumerate(test_file['pfirst_median'])]
    test_file['all2'] = [test_file['pfs'][indx] / (0.5*(test_file['pfirst'][indx]+test_file['psecond'][indx]))
                         for indx, _ in enumerate(test_file['pfirst_median'])]
    arr = []
    for indx, _ in enumerate(test_file['all']):
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
    arr1 = []
    for indx, _ in enumerate(test_file['all2']):
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
    return "final_GUI.csv", test_file['result'].mean(), test_file['result2'].mean()
