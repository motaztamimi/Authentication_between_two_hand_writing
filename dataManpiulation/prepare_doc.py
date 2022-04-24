from PIL import Image
import os
from matplotlib import pyplot as plt


def bigger_than(str1):
    temp1 = str1.split('-')
    return int(temp1[0])


def cut_width(page, page_num):
    '''
    Cut uncessery scanned page from left and right
    Note: there is a difference between pages
    '''
    width, height = page.size
    bottom = height
    right = width
    top = 0
    left = 0
    if page_num == 1:
        left = 150
        bottom = height - 100
        right = width - 50
    elif page_num == 2:
        right = width - 100
        bottom = bottom - 300

    cropped = page.crop((left, top, right, bottom))
    return cropped


def get_concat_vertical(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def prepare_doc_test(path,foleder = "data1_as_one_page"):
    files = os.listdir(path)
    files.sort(key=bigger_than)
    file_number = 1
    i = 0
    for file in files:
        i += 1
        img = Image.open(path + "/" + str(file))
        # file = os.path.splitext(file)[0] + ".jpeg"

        array_images = []
        try:
            page1 = cut_width(img, 1)
            array_images.append(page1)
            img.seek(1)
            page2 = cut_width(img, 2)
            array_images.append(page2)
        except EOFError:
            break

        # Create target Directory if don't exist
        if file_number == 1:
            concat_1 = get_concat_vertical(page1, page2)
            path_1 = '{}/{}.jpeg'.format(foleder, file.split('.')[0])
            concat_1.save(
                '{}/{}.jpeg'.format(foleder,file.split('.')[0]))
            file_number = 2
        else:
            concat_2 = get_concat_vertical(page1, page2)
            path_2 = '{}/{}.jpeg'.format(foleder,file.split('.')[0])
            concat_2.save(
                '{}/{}.jpeg'.format(foleder,file.split('.')[0]))

    return path_1, path_2
