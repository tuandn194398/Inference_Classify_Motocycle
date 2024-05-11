import csv
import os
import shutil

csv_path = 'data_color_segment/8210bb_segment1234.csv'
folder_path = 'data/valid/images'
file_list = []
arr_csv = []

with open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    # print('reader', reader)
    for row in reader:
        try:
            shutil.copy(os.path.join('data_color_segment/motorcycle/', row[0]), 'data_color_segment/13kbb_segment1234/')
            print('path:', os.path.join('data_color_segment/motorcycle/', row[0]))
        except:
            continue
        # arr_csv.append([row[0], row[1]])
    # for file in os.listdir(folder_path):

    # arr_csv.pop(0)

    # print(arr_csv[0:10])
