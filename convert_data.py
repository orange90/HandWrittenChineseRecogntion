import os
import numpy as np
import struct
from PIL import Image
import tensorflow as tf
import cv2

data_dir = ''
train_data_dir = os.path.join(data_dir, 'HWDB1.1trn_gnt')
test_data_dir = os.path.join(data_dir, 'HWDB1.1tst_gnt')


def get_file_count(data_dir):
    sub_folder = [data_dir]
    count = 0
    while len(sub_folder) > 0:
        for root, sub_folder, file_list in os.walk(data_dir):
            count += len(file_list)
            for sf in sub_folder:
                count += get_file_count(sf)
    return count


def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            if header_size + width*height != sample_size:
                break
            try:
                image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            except:
                print struct.pack('>H', tagcode).decode('gb2312')
            # print image, tagcode
            yield image, tagcode

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode


file_count_train = get_file_count('HWDB1.1trn_gnt')
print file_count_train

char_set = set()
for _, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    # try:
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    # print tagcode_unicode
    char_set.add(tagcode_unicode)
    # except:
    #     print 'unpack char %s failed'%train_data_dir
    #     with open('error_list.txt','a') as f:
    #         f.write(tagcode_unicode+'\n')


#save char list
char_list = list(char_set)
char_dict = dict(zip(sorted(char_list), range(len(char_list))))
print len(char_dict)
import pickle
f = open('char_dict', 'wb')
pickle.dump(char_dict, f)
f.close()
train_counter = 0
test_counter = 0



print 'start extracting training data'

file_count_train = get_file_count('HWDB1.1trn_gnt')
print file_count_train

for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    im = Image.fromarray(image)
    dir_name = './data/train/' + '%0.5d'%char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    im.convert('RGB').save(dir_name+'/' + str(train_counter) + '.png')
    train_counter += 1
    print '%d/%d'%(train_counter,file_count_train*3755)


print 'start extracting testing data'
file_count_test = get_file_count('HWDB1.1tst_gnt')
print file_count_test
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    im = Image.fromarray(image)
    dir_name = './data/test/' + '%0.5d'%char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    im.convert('RGB').save(dir_name+'/' + str(test_counter) + '.png')
    test_counter += 1
    print '%d/%d' % (train_counter, file_count_test*3755)