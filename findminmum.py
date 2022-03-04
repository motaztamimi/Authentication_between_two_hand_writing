import os
import cv2

def find_min():

    max_height = 9000
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
    print(toreturn)                
    os.remove(toreturn)                
    return max_height





if __name__ == "__main__":
    min = 1
    count =0;
    while min < 55:
       min = find_min()
       count+=1
    print(count)   

