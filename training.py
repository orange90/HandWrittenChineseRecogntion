# from read_data import *
import tensorflow as tf
# from tensorflow.contrib.keras.python.keras.preprocessing import image
# from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

# from kerlayers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
if tf.__version__ == '1.2.0':
    from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Dropout, Activation, Flatten, Dense
    from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
    from tensorflow.contrib.keras.api.keras.models import Sequential
    from tensorflow.contrib.keras.python.keras.utils import np_utils
else:
    from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, \
        GlobalAveragePooling2D, Dropout, Activation, Flatten, Dense
    from keras.layers.normalization import BatchNormalization
    from keras.models import Sequential
    from keras.utils import np_utils


# from fast_read_data import ChineseWrittenChars
#
# chars = ChineseWrittenChars()

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))
    return model

import numpy as np
# img_path = 'African_Bush_Elephant.jpg'
with tf.name_scope('model'):
    model = build_model()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

number_of_classes = 10

y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)

X_train/=255
X_test/=255
print X_train.shape

model.compile(optimizer='adam',loss='categorical_crossentropy')
model.summary()
model.fit(X_train, y_train,epochs=10)

y_pred = model.predict(X_test)
print (accuracy_score(y_test, y_pred))

# print('Predicted:', decode_predictions(preds, top=3)[0])

