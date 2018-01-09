import os
import tensorflow as tf
import random
from sklearn.utils import shuffle
import numpy as np
import struct
from PIL import Image

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()


ONLY_USE_100_CHARS = True
CHARS_100_LIST = None
CHARS_100_LIST_BINARIZED = None
if CHARS_100_LIST is None:
    CHARS_100_LIST = random.sample(range(0,3755),100)
    print 'selected chars are:',CHARS_100_LIST
    lb.fit(CHARS_100_LIST)
    CHARS_100_LIST_BINARIZED = lb.transform(CHARS_100_LIST)


class DataSet:
    global ONLY_USE_100_CHARS
    global CHARS_100_LIST
    global lb

    def __init__(self, path):
        self.iter_index = 0
        if ONLY_USE_100_CHARS:
            self.files, self.labels = self.get_100_input_lists(path)
        else:
            self.files, self.labels = self.get_all_input_lists(path)
        self.files, self.labels = shuffle(self.files, self.labels)

    def get_100_input_lists(self, data_dir):
        image_names = []
        label_list = []
        for root, sub_folder, file_list in os.walk(data_dir):
            # print root, sub_folder, file_list
            try:
                test_num = int(root[-5:])
            except:
                print 'this is not a data folder, skipped'
                continue
            if test_num in CHARS_100_LIST:
                image_names += [os.path.join(root, file_path) for file_path in file_list]
                for i in range(len(file_list)):
                    label_list.append(test_num)
        return image_names, label_list

    @staticmethod
    def read_from_gnt_dir(gnt_dir):
        def one_file(f):
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
                except:
                    print struct.pack('>H', tagcode).decode('gb2312')
                yield image, tagcode

        for file_name in os.listdir(gnt_dir):
            if file_name.endswith('.gnt'):
                file_path = os.path.join(gnt_dir, file_name)
                with open(file_path, 'rb') as f:
                    for image, tagcode in one_file(f):
                        yield image,tagcode


    def get_all_input_lists(self, data_dir):
        image_names = []
        label_list = []
        label_len = []
        for root, sub_folder, file_list in os.walk(data_dir):
            image_names += [os.path.join(root, file_path) for file_path in file_list]
            for name in sub_folder:
                current_dir = os.path.join(root, name)
                set_images = os.listdir(current_dir)
                label_len.append(len(set_images))
        index = 0
        while index < (len(label_len)):
            i = 0
            while i in range(label_len[index]):
                label_list.append(index)
                i += 1
            index += 1
        return image_names, label_list

    def load_next_batch(self, batch_size=200,flatten=False):
        for image, tagcode in self.read_from_gnt_dir(gnt_dir=train_data_dir):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            im = Image.fromarray(image)
            dir_name = './data/train/' + '%0.5d' % char_dict[tagcode_unicode]
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            im.convert('RGB').save(dir_name + '/' + str(train_counter) + '.png')
            train_counter += 1
            print '%d/%d' % (train_counter, file_count_train * 3755)
        x = []
        y = []
        with tf.Session() as sess:
            for i in range(batch_size):
                index = self.iter_index + i
                oper = self.read_convert_image(self.files[index])
                # result = sess.run(oper)
                # if flatten:
                #     result = result.flatten
                x.append(oper)
                y.append(self.labels[index])
            x = sess.run(x)
            x = np.array(x)
            if flatten:
                x = x.flatten().reshape(batch_size,-1)
        self.iter_index += batch_size
        x = np.array(x)
        y = np.array(y)
        y = lb.transform(y)
        return x, y

    def load_all(self, flatten=False):
        all_size = len(self.labels)
        x,y = self.load_next_batch(all_size,flatten)
        self.iter_index = 0
        return x,y

    def read_convert_image(self, image_name):
        images_content = tf.read_file(image_name)
        image = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        new_size = tf.constant([64, 64], dtype=tf.int32)
        new_image = tf.image.resize_images(image, new_size)
        return new_image


class ChineseWrittenChars:
    def __init__(self):
        self.train = DataSet('HWDB1.1trn_gnt')
        self.test = DataSet('HWDB1.1tst_gnt')
        print 'inited datasets successfully'
#
# loader = ChineseWrittenChars()
# import time
# start_time = time.time()
# result,y = loader.train.load_next_batch(batch_size=10)
# print time.time()-start_time
# print result,y
# result,y = loader.train.load_next_batch(batch_size=10)
# print result,y