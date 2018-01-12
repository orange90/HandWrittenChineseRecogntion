import os
import tensorflow as tf
import random
from sklearn.utils import shuffle
import numpy as np
import struct
from PIL import Image
import pickle
import time


class DataSet:

    def __init__(self, path, is_train=True, char_dict=None):
        self.file_counter = 0
        self.is_train = is_train
        self.iter_index = 0
        self.path = path
        self.char_dict = char_dict

    def one_file(self, f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            try:
                image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                # image = Image.fromarray(image)
            except:
                print struct.pack('>H', tagcode).decode('gb2312')
            yield image, tagcode

    def read_from_gnt_dir(self, gnt_dir):
        for file_name in os.listdir(gnt_dir):
            if file_name.endswith('.gnt'):
                file_path = os.path.join(gnt_dir, file_name)
                with open(file_path, 'rb') as f:
                    for image, tagcode in self.one_file(f):
                        yield image,tagcode

    def read_one_gnt_file(self):
        for file_name in os.listdir(self.path):
            if file_name.endswith('.gnt'):
                file_path = os.path.join(self.path, file_name)
                with open(file_path, 'rb') as f:
                    x = []
                    y = []
                    for image, tagcode in self.one_file(f):
                        x.append(image)
                        y.append(tagcode)
                yield x, y

    #todo: to unittest
    def load_next_file(self):
        for x_s,y_s in self.read_one_gnt_file():
            # with tf.Session() as sess:
            result_x = []
            result_y = []
            for i in range(len(x_s)):
                result = self.read_convert_image(x_s[i])
                result_x.append(result)
                result_y.append(y_s[i])
            x = np.array(result_x)
            y = np.array(result_y)
            self.file_counter += 1
            print ('loaded files count',self.file_counter)
            yield x, y

    def load_all(self):
        all_size = len(os.listdir(self.path))
        x = []
        y = []
        for temp_x,temp_y in self.load_next_file():
            x.extend(temp_x)
            y.extend(temp_y)
        return np.array(x), np.array(y)


    def read_convert_image(self, image):
        im = Image.fromarray(image)
        im = im.resize([64,64])
        new_image = np.asarray(im)
        new_image = new_image.reshape(new_image.shape[0], new_image.shape[1], 1)
        # print type(new_image)
        # im.show()
        # image_tensor = tf.image.convert_image_dtype(image, tf.float32)
        # new_size = tf.constant([64, 64], dtype=tf.int32)
        # new_image = tf.image.resize_images(image_tensor, new_size)
        return new_image


class ChineseWrittenChars:
    def __init__(self):
        self.train = DataSet('HWDB1.1trn_gnt',is_train=True)

        # self.char_dict = self.generate_char_dict()
        print('--------create char_dict successfully------------')

        self.test = DataSet('HWDB1.1tst_gnt',is_train=False)
        print 'inited datasets successfully'

    def generate_char_list(self):
        if os.path.isfile('char_list'):
            with open('char_list', 'rb') as f:
                print 'char dict had been generated, just load'
                char_list = pickle.load(f)
                return char_list
        else:
            char_list = []
            for _, tagcode in self.train.read_from_gnt_dir(gnt_dir='HWDB1.1trn_gnt'):
                char_list.append(tagcode)
            with open('char_list', 'wb') as f:
                pickle.dump(char_list,f)
            return char_list

#
# loader = ChineseWrittenChars()
# import time
# start_time = time.time()
# result,y = loader.train.load_next_batch(batch_size=10)
# print time.time()-start_time
# print result,y
# result,y = loader.train.load_next_batch(batch_size=10)
# print result,y