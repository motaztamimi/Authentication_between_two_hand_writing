import detection_function
import cv2
import prepare_doc
from PIL import Image
import os
import pandas as pd


def from_two_pages_to_jpeg(path='data1_as_one_page'):

    if not os.path.exists(path):
        os.mkdir(path)
        print('change the file type from 2 pages of type tiff to one jpeg image...')
        prepare_doc.prepare_doc_test(
            '/Users/mustafasmac/Desktop/4th_year/final_project/data1')

        print('Done.')


def creating_lines_for_each_person(path='data1_as_one_page'):

    if os.path.exists(path):

        print('Finding lines for each person...')

        files = os.listdir(path)
        path_1 = 'data_for_each_person'
        if not os.path.exists(path_1):
            os.mkdir(path_1)
        for file in files:
            if file != '.DS_Store':
                number = int(file.split('-')[0])
                dir_name = 'person_{}'.format(number)
                if not os.path.exists(os.path.join(path_1, dir_name)):
                    os.mkdir(path_1 + '/' + dir_name)
                img = cv2.imread('data1_as_one_page' + '/' + file, 0)
                lines = detection_function.detect_lines(img)
                for indx, line in enumerate(lines):
                    to_save = Image.fromarray(line)
                    name = file.split('-')[1][0]
                    name = 'p' + str(number) + '_' + name + \
                        '_L_' + str((indx + 1))
                    to_save.save('{}/{}/{}.jpeg'.format(path_1,
                                                        dir_name, name))

        print('Done.')


def find_match_pairs(path='data_for_each_person'):

    if os.path.exists(path):

        print('Creating all possible pairs of lines that creat a Match and store the labels in csv file...')

        genuin_data = []

        dir_name = path
        dirs = os.listdir(dir_name)

        for _dir in dirs:
            if _dir != '.DS_Store':
                files = os.listdir(dir_name + '/' + _dir)
                for indx, img1 in enumerate(files):
                    for img2 in files[indx:]:
                        to_add = []
                        if img1 != img2:
                            to_add.append(dir_name + '/' + _dir + '/' + img1)
                            to_add.append(dir_name + '/' + _dir + '/' + img2)
                            to_add.append('0')
                            genuin_data.append(to_add)

        csv_file = pd.DataFrame(genuin_data)
        csv_file.to_csv('test.csv', index=False, sep=',', header=0)

        print('Done.')


def find_miss_match_pairs(path='data_for_each_person'):

    if os.path.exists(path):

        print('Creating all pairs of lines that creat a Miss Match and store the labels in csv file...')

        diff_data = []
        dir_name = path
        dirs = os.listdir(dir_name)

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
                                    to_add.append(dir_name + '/' +
                                                  _dir + '/' + img1)
                                    to_add.append(dir_name + '/' +
                                                  _dir_ + '/' + img2)
                                    to_add.append('1')
                                    diff_data.append(to_add)

        csv_file = pd.DataFrame(diff_data)
        csv_file.to_csv('test1.csv', index=False, sep=',', header=0)

        print('Done.')


if __name__ == '__main__':
    print('Starting the preparing phase...')
    from_two_pages_to_jpeg()
    creating_lines_for_each_person()
    find_match_pairs()
    find_miss_match_pairs()
    print('Done. Now you can use the data')
