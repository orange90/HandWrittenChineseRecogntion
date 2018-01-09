from read_data import *
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.preprocessing import image
from tensorflow.contrib.keras.python.keras.applications.resnet50 import ResNet50,preprocess_input, decode_predictions
from keras.datasets import mnist
from sklearn.metrics import accuracy_score


import numpy as np
# img_path = 'African_Bush_Elephant.jpg'
with tf.name_scope('model'):
    model = ResNet50(include_top=False, weights=None, input_shape=(28,28,3))
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

(X_train, y_train),(X_test, y_test) = mnist.load_data()
print X_test.shape

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train, y_train,epochs=10)
model.fit
y_pred = model.predict(X_test)
print (accuracy_score(y_test, y_pred))

# print('Predicted:', decode_predictions(preds, top=3)[0])