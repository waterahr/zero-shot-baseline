"""
python feature_extractor.py --name inceptionv3 --image_size 139 --weights
"""
import pandas as pd
import numpy as np
import math
#build model
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, Sequential
from keras.applications import VGG16, VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
import os
import argparse
import pickle
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def parse_arg():
    parser = argparse.ArgumentParser(description='training of the baseline...')
    parser.add_argument('--gpus', type=str, default='',
                        help='gpu device\'s ID need to be used')
    parser.add_argument('--name', type=str, default='',
                       help='the model has been learned, including : ' + str(model_names))
    parser.add_argument('--image_size', type=int, default=64,
                       help='the image size need to input the network')
    parser.add_argument('--weights', type=str, default='',
                       help='the image size need to input the network')
    args = parser.parse_args()
    if args.name not in model_names:
        print("Try again!Input the right model name as following: ")
        print(model_names)
        exit()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

model_names = ['inceptionv3', 'vgg19', 'mobilenet', 'densenet121', 'resnet50']
args = parse_arg()
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

#Loading the dataset as a dictionary: (img_name, label)
data_train = open(r'/home/anhaoran/data/zero-shot-tianchi/dataset_B/DatasetB_20180919/train.txt')
data_train = data_train.readlines()
data_dict = {}
length = len(data_train)
for i in range(length):
    m,n = data_train[i].split()
    data_dict.update({m : n})
print("The number of train samples is : ", len(data_dict))

data_test = open("/home/anhaoran/data/zero-shot-tianchi/dataset_B/DatasetB_20180919/image.txt")
data_test = data_test.readlines()#[:2]
length = len(data_test)
for i in range(length):
    data_dict.update({data_test[i][:-1] : "test"})
print("The number of all samples is : ", len(data_dict))

#print("The data and their label is : \n", data_dict)

#Loading the images as a array(data_x), img_name as a list(images_all), labels as a list(labels_all)
image_size = args.image_size


images_all = list()
labels_all = list()
path_test = r'/home/anhaoran/data/zero-shot-tianchi/dataset_B/DatasetB_20180919/test/'
path_train = r'/home/anhaoran/data/zero-shot-tianchi/dataset_B/DatasetB_20180919/train/'
length = len(data_dict)
data_x = np.zeros((length, image_size, image_size, 3))
idx = 0
for img_name, img_label in data_dict.items():
    if img_label != "test":
        img = image.load_img(path_train + img_name, target_size=(image_size, image_size, 3))
    else:
        img = image.load_img(path_test + img_name, target_size=(image_size, image_size, 3))
    data_x[idx] = image.img_to_array(img)
    images_all.append(img_name)
    labels_all.append(img_label)
    idx += 1

print("The data's shape is : ", data_x.shape)
print("The length of images_all is : ", len(images_all))
#print(images_all)
print("The length of labels_all is : ", len(labels_all))
#print(labels_all)


#Loading the model trained
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

model_dir = ""
model_name = ""
if args.name == "inceptionv3":
    model_dir = "../models/inceptionv3/"
    model_name = "Inceptionv3a25DataB"
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(image_size, image_size, 3))
elif args.name == "vgg19":
    model_dir = "../models/vgg19/"
    model_name = "vgg19pre_"
    base_model = VGG19(include_top=False, weights=None, input_shape=(image_size, image_size, 3))
elif args.name == "densenet121":
    model_dir = "../models/densenet121/"
    model_name = "DenseNet121c160"
    base_model = DenseNet121(weights=None, include_top=False, pooling = 'avg')
elif args.name == "mobilenet":
    model_dir = "../models/mobilenet/"
    model_name = "mobile"
    base_model = MobileNet(weights=None, include_top=False, pooling = 'avg')
elif args.name == "resnet50":
    model_dir = "../models/resnet50/"
    model_name = "resnet50"
    base_model = ResNet50(weights=None, include_top=False, pooling = 'avg')

x = base_model.output
x = Flatten()(x)
#x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', name="dense_feature")(x)
model = Model(inputs=base_model.input,
                  outputs=x)
"""
base_model = MobileNet(include_top=True, weights=None,
                           input_tensor=None, input_shape=None,
                           pooling=None, classes=40)
base_model.load_weights("/home/anhaoran/codes/zero-shot-cnn/Baselines/zero_shot_learning_baseline/model/mobile_Animals_wgt.h5")
model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('global_average_pooling2d_1').output)
"""
if args.weights == '':
    model.load_weights(model_dir + model_name + "_baseline_model.h5", by_name=True)
else:
    model.load_weights(args.weights, by_name=True)
model.summary()

#Predicting the features
features_all = model.predict(data_x)
images_all = images_all
labels_all = labels_all
data_all = {'features_all':features_all, 
            'labels_all':labels_all,
            'images_all':images_all}
savename = "../results/" + model_name + '_features_all.pickle'
fsave = open(savename, 'wb')
pickle.dump(data_all, fsave)
fsave.close()
print("The features are :\n", features_all)
