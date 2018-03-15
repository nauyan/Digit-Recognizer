import numpy as np
import csv
from skimage import io
import progressbar
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
from keras.initializers import constant
from skimage import io, exposure
import tensorflow

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.layers.advanced_activations import LeakyReLU
from glob import glob


from keras import backend as K

os.environ['KERAS_BACKEND'] = 'tensorflow'
bar = progressbar.ProgressBar()


idx=0
immatrix =np.zeros(shape=(42000,28,28,1))
labels =np.zeros(shape=(42000))
with open('train.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile)
     next(spamreader, None)
     for row in bar(spamreader):
        img=row[1:]
        img=np.asarray(img)
        img=img.reshape(28,28,1)
        immatrix[idx]=img
        labels[idx]=row[0]
        #print(img.shape)
        #io.imsave('Test//{}_{}.png'.format(row[0], idx), img)
        
        idx=idx+1
print(immatrix.shape)
print(labels.shape)   
#print(np.unique(labels))
train_data = [immatrix,labels] 

batch_size = 25

num_classes = 10

nb_epoch = 12    

X_train, X_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size=0.2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()

#First convolution Layer
model.add(Convolution2D(256, (3, 3), padding='same', input_shape=(28,28,1))) #Try for 25x25 patch
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.10))

model.add(Convolution2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.20))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.30))

model.add(Flatten()) # No dropout after flattening.
model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = SGD(lr=0.01)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics = ['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=1,shuffle=True, validation_data=(X_test, Y_test))
pandas.DataFrame(hist.history).to_csv("T1Patches.csv")     


train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)


from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
  

y_pred = model.predict_classes(X_test)
print(y_pred)

p=model.predict_proba(X_test) # to predict probability

target_names = ['class 0(normal)', 'class 1(necros)', 'class 2(adema)','class 3(enhancing)','class 4(non-enhancing)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# saving weights

fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)
       
