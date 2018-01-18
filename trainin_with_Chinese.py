import time
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from fast_read_data import ChineseWrittenChars

chars = ChineseWrittenChars()
chars.test.use_rotation = False
chars.test.use_filter = False

lb = LabelBinarizer()
lb.fit(chars.generate_char_list())
number_of_classes = 3755

def build_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 1)))
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
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(number_of_classes))

    model.add(Activation('softmax'))
    return model


def training(X_train,y_train):
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=2) # the more epoch the better
    model.save('model.h5')


def testing(X_test, y_test):
    # load model
    from keras.models import load_model
    model = load_model('model.h5')
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    loaded_model_score = model.evaluate(X_test, y_test)
    print('test accuracy: ',loaded_model_score[1]) # the 0-th element is loss, the 1st element is accuracy
    print(model.metrics)


def app(train_or_test):
    if train_or_test == 'train':
        start_time = time.time()
        X_train, y_train = chars.train.load_all()
        X_train /= 255
        y_train = lb.transform(y_train)
        print 'load training used time:', time.time() - start_time
        print X_train.shape
        print y_train.shape
        training(X_train, y_train)

    if train_or_test == 'test':
        X_test, y_test = chars.test.load_all()
        y_test = lb.transform(y_test)
        X_test /= 255
        testing(X_test, y_test)

t

