import csv
import pandas as pd
import os
import re

def return_highest_num_from_list(_list):
    start_num = 0
    for file in _list:
        num = int(re.findall('\d+', file)[0])
        if num > start_num:
            start_num = num

    return start_num

def main_1():
    print(os.getcwd())
    base = './../../mart_pics/'
    pd_dat = pd.read_csv(base + 'Boxesmartin.csv')

    start_book = return_highest_num_from_list(os.listdir('./../../pics_/books'))
    start_box = return_highest_num_from_list(os.listdir('./../../pics_/boxes'))
    start_cup = return_highest_num_from_list(os.listdir('./../../pics_/cups'))

    print(start_box)
    print(start_book)
    print(start_cup)
    base_go_to = './../../pics_/'

    for i in range(len(pd_dat['ImgPath'])):
        img_path = str(pd_dat['ImgPath'][i]) + '.jpg'
        img_path = img_path[0:]
        print(img_path)
        class_name = pd_dat['ClassName'][i]

        print(class_name)
        if 'Book' in class_name:
            start_book += 1
            os.rename(base + img_path, base_go_to + 'books/' + f'{start_book}.jpg')
        elif 'Box' in class_name:
            start_box += 1
            os.rename(base + img_path, base_go_to + 'boxes/' + f'{start_box}.jpg')
        elif 'Coffee cup' in class_name:
            start_cup += 1
            os.rename(base + img_path, base_go_to + 'cups/' + f'{start_cup}.jpg')


def make_csv():
    path_img = './../../pics_/'

    with open('./../../pics_/list.csv', 'w+', newline='') as file:
        writer = csv.writer(file)

        for img in os.listdir(path_img + 'books'):
            writer.writerow([f'./../pics_/books/{img}', 'Book'])

        for img in os.listdir(path_img + 'cups'):
            writer.writerow([f'./../pics_/cups/{img}', 'Coffee cup'])

        for img in os.listdir(path_img + 'boxes'):
            writer.writerow([f'./../pics_/boxes/{img}', 'Box'])


if __name__ == '__main__':
    # main_1()
    make_csv()