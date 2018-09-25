"""
python train_model_attribute_pre.py --gpus 1 --name inceptionv3 --image_size 64 --batch_size 64
"""
import pandas as pd
import numpy as np
import math
import random
from keras.layers import Conv2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, Sequential
from keras.applications import VGG16, VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras.utils import multi_gpu_model
import os
import datetime
import argparse
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def array2dic(array, classes):
    index = 0
    dic = {}
    for m,n in classes.items():
        old_index = index
        index += n
        dic.update({m : array[:, old_index:index]})
    return dic

def generate_batch_data(X, y, batch_size, classes):
    length = len(y)
    index = [i for i in range(0, length // batch_size + 1)]
    random.shuffle(index)
    for i in range(len(index)):
        start_idx = index[i] * batch_size
        end_idx = index[i+1] * batch_size
        if end_idx >= length:
            end_idx = length
        yield X[start_idx:end_idx], array2dic(y[start_idx:end_idx], classes)


def parse_arg():
    parser = argparse.ArgumentParser(description='training of the baseline...')
    parser.add_argument('--gpus', type=str, default='',
                        help='gpu device\'s ID need to be used')
    parser.add_argument('--name', type=str, default='',
                       help='the model has been learned, including : ' + str(model_names))
    parser.add_argument('--image_size', type=int, default=32,
                       help='the image size need to input the network')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='the batch size need to train the model')
    parser.add_argument('--weights', type=str, default='',
                       help='the weights file need to train the model')
    args = parser.parse_args()
    if args.name not in model_names:
        print("Try again!Input the right model name as following: ")
        print(model_names)
        exit()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

model_names = ['inceptionv3', 'vgg19', 'mobilenet', 'densenet121', 'resnet50']
args = parse_arg()
gpus_num = len(args.gpus.split(','))

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

model_dir = ""
model_name = ""
if args.name == "inceptionv3":
    model_dir = "../models/inceptionv3/"
    model_name = "inceptionv3a25"
elif args.name == "vgg19":
    model_dir = "../models/vgg19/"
    model_name = "vgg19_2018-08-27"
elif args.name == "densenet121":
    model_dir = "../models/densenet121/"
    model_name = "DenseNet121"
elif args.name == "mobilenet":
    model_dir = "../models/mobilenet/"
    model_name = "mobile"
elif args.name == "resnet50":
    model_dir = "../models/resnet50/"
    model_name = "resnet50"
image_size = args.image_size
batch_size = args.batch_size
nb_epoch = 200

def schedule_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


#Define the callbacks: checkpointer, decaylr, reducelr, earlystop, csvlog
"""
monitor = 'loss'
checkpointer = ModelCheckpoint(filepath = model_dir + model_name + '_epoch{epoch:02d}_valloss{'+ monitor + ':.2f}.hdf5',
                               monitor = monitor,
                               verbose=1, 
                               save_best_only=True, 
                               save_weights_only=False,
                               mode='auto', 
                               period=1)
                               """
monitor = 'val_loss'
checkpointer = ModelCheckpoint(filepath = model_dir + model_name + '_epoch{epoch:02d}_valloss{'+ monitor + ':.2f}.hdf5',
                               monitor = monitor,
                               verbose=1, 
                               save_best_only=True, 
                               save_weights_only=False,
                               mode='auto', 
                               period=25)
decaylr = LearningRateScheduler(schedule_decay)
reducelr = ReduceLROnPlateau(monitor = monitor,
                             factor=0.1, 
                             patience=4, 
                             verbose=0, 
                             mode='auto', 
                             epsilon=0.0001, 
                             cooldown=0,
                             min_lr=0)
earlystop = EarlyStopping(monitor= monitor, patience=20, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir=model_dir + 'logs', 
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=False, 
                          embeddings_freq=0,
                          embeddings_layer_names=None, 
                          embeddings_metadata=None)
csvlog = CSVLogger(model_dir + 'logs/log_attributes.csv')


#Define the model used to trainning without top layer -> feature extracting mainly used future
#'imagenet' -> None
classes = { 'animal' : 1,
           'transportation' : 1,
           'clothes' : 1,
           'plant' : 1,
           'tableware' : 1,
           'device' : 1,
           'other_classes' : 1,
           'black' : 1,
           'white' : 1,
           'blue' : 1,
           'brown' : 1,
           'orange' : 1,
           'red' : 1,
           'green' : 1,
           'yellow' : 1,
           'has_feathers' : 1,
           'has_four_legs' : 1,
           'has_two_legs' : 1,
           'has_two_arms' : 1,
           'for_entertainment' : 1,
           'for_business' : 1,
           'for_communication' : 1,
           'for_family' : 1,
           'for_office_use' : 1,
           'for_personal' : 1
}    
"""
    'cla' : 6+1,
  #  'clo' : 8, #暂且不用
    'has' : 4,
    'for' : 6
  #  'is' : 6 #暂且不用
    'gorgeous' : 1,
    'simple' : 1,
    'elegant' : 1,
    'cute' : 1,
    'pure' : 1,
    'naive' : 1
"""
if args.name == "inceptionv3":
    #base_model = InceptionV3(weights=None, include_top=False, pooling = 'avg')
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(image_size, image_size, 3))
elif args.name == "vgg19":
    #base_model = VGG19(weights=None, include_top=False, pooling = 'avg')
    base_model = VGG19(include_top=False, weights=None, input_shape=(image_size, image_size, 3))
elif args.name == "densenet121":
    base_model = DenseNet121(weights=None, include_top=False, pooling = 'avg')
    #base_model = DenseNet121(include_top=False, weights=None, input_shape=(image_size, image_size, 3))
elif args.name == "mobilenet":
    base_model = MobileNet(weights=None, include_top=False, pooling = 'avg')
    #base_model = MobileNet(include_top=False, weights=None, input_shape=(image_size, image_size, 3))
elif args.name == "resnet50":
    base_model = ResNet50(weights=None, include_top=False, pooling = 'avg')
    #base_model = ResNet50(include_top=False, weights=None, input_shape=(image_size, image_size, 3))

encoder = base_model.output
if args.name == "inceptionv3":
    """
    x = Convolution2D(2048, 5, 5, activation='relu', border_mode='valid')(encoder)
    x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)#2
    x = Convolution2D(768, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)#4
    x = Convolution2D(768, 3, 3, activation='relu', border_mode='valid')(x)
    x = UpSampling2D((2, 2))(x)#4
    x = Convolution2D(384, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)#8
    x = Convolution2D(384, 2, 2, activation='relu', border_mode='valid')(x)
    x = UpSampling2D((2, 2))(x)#14
    x = Convolution2D(288, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)#28
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)#56
    x = Convolution2D(192, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(80, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)#112
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)#224
    decoder = Convolution2D(3, 224, 1, activation='relu', border_mode='same')(x)
    """
    decoder = x
elif args.name == "vgg19":
    x = base_model.output
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), strides=(1, 1), padding='same')(x)
    decoder = Conv2D(3, (3, 3), strides=(1, 1), padding='same')(x)

model = Model(base_model.input, decoder)

if gpus_num != 1:
    
    #with tf.device("/cpu:0"):
        #model = Model(inputs=base_model.input, outputs= predictions)
    model = multi_gpu_model(model, gpus=gpus_num)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy')


#Loading the dataset: train_x(each row as a image array) and train_y(attributes per image)
data_atten = pd.read_csv(r'../test.csv')
data_atten = data_atten.set_index('0')
data_train = open(r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/train.txt')
data_train = data_train.readlines()
print(data_train[0])
path = r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/train/'
length = len(data_train)
train_x = np.zeros((length, image_size, image_size, 3))
train_y = np.zeros((length, 25))
for i in range(length):
    m,n = data_train[i].split()
    #img = image.load_img(path + m)
    img = image.load_img(path + m, target_size=(image_size, image_size, 3))
    train_x[i] = image.img_to_array(img) / 255.
    train_y[i] = data_atten.loc[n]
       
# Data augmentation to pre-processing
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

print("The shape of the X_train is: ", train_x.shape)
print("The shape of the y_train is: ", train_y.shape)
    

#Train the model

#model.load_weights(model_dir + 'inceptionv3_2018-08-26_epoch50_7.30.hdf5')
"""
#"Deal with the train_y to a dictionnary
y_train = array2dic(train_y, classes)
if args.weights != '':
    model.load_weights(args.weights, by_name=True)
model.fit(train_x, train_x,
         epochs = nb_epoch,
         batch_size = batch_size,
         validation_split = 0.3,
         callbacks = [checkpointer, csvlog])
"""
X_train, X_test, y_train, y_test = train_test_split(train_x,train_y, test_size=0.3, random_state=0)
#"""
#Deal with the train_y to a dictionnary
#y_train = array2dic(y_train, classes)
#y_test = array2dic(y_test, classes)
train_generator = datagen.flow(X_train, X_train, batch_size=batch_size)#--->must ndarray
val_generator = datagen.flow(X_test, X_test, batch_size=batch_size)#--->must ndarray

model.fit_generator(train_generator,
            steps_per_epoch = int(X_train.shape[0] / (batch_size * gpus_num)),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test.shape[0] / (batch_size * gpus_num)),
            callbacks = [checkpointer, csvlog])
#"""
model.save(model_dir + model_name + '_pre_baseline_model.h5')
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
os.system('mv nohup.out ' + model_dir + 'nohup_' + now_time + '.out')
