import pandas as pd
import numpy as np
import math
import random
#build model
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, Sequential
from keras.applications import VGG16, VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger

#model_dir = "./inceptionv3/"
#model_name = "inceptionv3"
model_dir = "./vgg19_without_attributes/"
model_name = "vgg19"
image_size = 64
batch_size = 64
nb_epoch = 100
nb_classes = 230

def schedule_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

checkpointer = ModelCheckpoint(filepath = model_dir + model_name + '_epoch{epoch:02d}_valloss{val_loss:.2f}.hdf5',
                               monitor = 'val_loss',
                               verbose=1, 
                               save_best_only=True, 
                               save_weights_only=False,
                               mode='auto', 
                               period=25)
decaylr = LearningRateScheduler(schedule_decay)
reducelr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.1, 
                             patience=4, 
                             verbose=0, 
                             mode='auto', 
                             epsilon=0.0001, 
                             cooldown=0,
                             min_lr=0)
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir=model_dir + 'logs', 
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=False, 
                          embeddings_freq=0,
                          embeddings_layer_names=None, 
                          embeddings_metadata=None)
csvlog = CSVLogger(model_dir + 'logs/log.csv')

base_model = VGG19(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax', name="class")(x)

model = Model(inputs=base_model.input, outputs= predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

data_labels = []
data_train = open(r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/train.txt')
data_train = data_train.readlines()
data_train = data_train
print(data_train[0])
path = r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/train/'
length = len(data_train)
train_x = np.zeros((length, image_size, image_size, 3))
train_y = np.zeros((length, nb_classes))
for i in range(length):
    m,n = data_train[i].split()
    #img = image.load_img(path + m)
    img = image.load_img(path + m, target_size=(image_size, image_size, 3))
    train_x[i] = image.img_to_array(img)
    if n not in data_labels:
        data_labels.append(n)
    train_y[i][data_labels.index(n)] = 1
print(len(data_labels))
    
# Use heavy augmentation if you plan to use the model with the
# accompanying webcam.py app, because webcam data is quite different from photos.
heavy_augmentation = True
if heavy_augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.5,
        channel_shift_range=0.5,
        fill_mode='nearest')
else:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')
datagen.fit(train_x)
index = [i for i in range(len(train_x))]  
random.shuffle(index) 
train_x = train_x[index]
train_y = train_y[index]
    
#np.save('abc.npy', train_x)
#train_x = np.load('abc.npy')
print("The shape of the X_train is: ", train_x.shape)
print("The shape of the y_train is: ", train_y.shape)

#"""
model.fit(train_x, train_y,
         epochs = nb_epoch,
         validation_split = 0.2,
         callbacks = [checkpointer, csvlog])
"""
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = datagen.flow(X_test, y_test, batch_size=batch_size)
model.fit_generator(train_generator,
            steps_per_epoch = int(X_train.shape[0] / batch_size),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test.shape[0] / batch_size),
            callbacks = [checkpointer, decaylr, reducelr, tensorboard, csvlog])
"""
model.save(model_dir + 'baseline_model.h5')