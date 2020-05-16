# Convolutional Neural Networks for Image Classification of 10 category
# using Kares /Tensorflow
# keras built in data set "fashion_mnist"

'''
Label    Description
0        T-shirt/top
1        Trouser
2        Pullover
3        Dress
4        Coat
5        Sandal
6        Shirt
7        Sneaker
8        Bag
9        Ankle boot
'''
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

plt.imshow(x_train[2]) # showing the image #2 of the data set as an example

x_train = x_train/255
x_test = x_test/255
# x is in shape of (60000, 28, 28) s owe need to reshape it
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)


y_cat_train = to_categorical(y_train) # to make them one-hot coded
y_cat_test = to_categorical(y_test) # to make them one-hot coded

################## Creating the CNN model ####################
model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu'))

# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
###############################################################

model.fit(x_train,y_cat_train,epochs=12)
acc= model.evaluate(x_test,y_cat_test)[1]
print('\n')
print('Accuracy of model on the test data is: ',acc)

from sklearn.metrics import classification_report
predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))
