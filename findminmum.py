import os
import cv2


def find_min():
     if os.path.exists("data_for_each_person"):

        print('Creating all possible pairs of lines that creat a Match and store the labels in csv file...')
     max_height=9000
     dir_name = "data_for_each_person"
     dirs = os.listdir(dir_name)
     for _dir in dirs:
         if _dir !='.DS_Store':
             files = os.listdir(dir_name + '/' + _dir)
             for indx , img1 in enumerate(files):
                imgpath = dir_name+"/"+_dir+"/"+img1
                img = cv2.imread(imgpath, 0)
                if img.shape[0] < max_height:
                    toreturn= imgpath
                    max_height=img.shape[0]
     print(max_height)
     print(toreturn)





find_min()


