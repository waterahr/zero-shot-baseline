import pandas as pd
import numpy as np
import math
#build model
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, Sequential
from keras.applications import VGG16, VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense
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
data_dict = {}
data_test = open("/home/anhaoran/data/zero-shot-tianchi/dataset_A/test/image.txt")
data_test = data_test.readlines()#[:2]
length = len(data_test)
for i in range(length):
    data_dict.update({data_test[i][:-1] : "test"})
print("The number of test samples is : ", len(data_dict))

#print("The data and their label is : \n", data_dict)

#Loading the images as a array(data_x), img_name as a list(images_all)
image_size = args.image_size

images_all = list()
path_test = r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/test/test/'
length = len(data_dict)
data_x = np.zeros((length, image_size, image_size, 3))
idx = 0
for img_name, img_label in data_dict.items():
    img = image.load_img(path_test + img_name, target_size=(image_size, image_size, 3))
    data_x[idx] = image.img_to_array(img)
    images_all.append(img_name)
    idx += 1

print("The data's shape is : ", data_x.shape)
print("The length of images_all is : ", len(images_all))
#print(images_all)
#print(labels_all)

#Loading the model trained
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet import MobileNet

model_dir = ""
model_name = ""
base_model = ""
predictions = ""
classes = ""
if args.name == "inceptionv3":
    model_dir = "../models/inceptionv3/"
    model_name = "inceptionv3_2018-08-28"
elif args.name == "vgg19":
    model_dir = "../models/vgg19/"
    model_name = "vgg19_2018-08-27"
elif args.name == "densenet121":
    model_dir = "../models/densenet121/"
    model_name = "DenseNet121c160"
    base_model = DenseNet121( weights = None, include_top = False, pooling='avg' )
    base_modelOutput = base_model.output
    #x = Flatten()( base_modelOutput )
    x = Dense(1024, activation = 'relu')( base_modelOutput )
    predictions = Dense(160, activation = None)( x )
    classes = ['ZJL133', 'ZJL198', 'ZJL11', 'ZJL117', 'ZJL176', 'ZJL84', 'ZJL116', 'ZJL25', 'ZJL36', 'ZJL142', 'ZJL79', 'ZJL158', 'ZJL130', 'ZJL12', 'ZJL60', 'ZJL97', 'ZJL39', 'ZJL125', 'ZJL16', 'ZJL47', 'ZJL72', 'ZJL44', 'ZJL110', 'ZJL191', 'ZJL6', 'ZJL186', 'ZJL107', 'ZJL164', 'ZJL120', 'ZJL179', 'ZJL151', 'ZJL78', 'ZJL2', 'ZJL193', 'ZJL24', 'ZJL50', 'ZJL34', 'ZJL127', 'ZJL43', 'ZJL23', 'ZJL146', 'ZJL89', 'ZJL138', 'ZJL103', 'ZJL175', 'ZJL195', 'ZJL94', 'ZJL185', 'ZJL154', 'ZJL126', 'ZJL200', 'ZJL98', 'ZJL56', 'ZJL22', 'ZJL124', 'ZJL73', 'ZJL52', 'ZJL42', 'ZJL159', 'ZJL13', 'ZJL67', 'ZJL63', 'ZJL81', 'ZJL77', 'ZJL101', 'ZJL167', 'ZJL66', 'ZJL190', 'ZJL160', 'ZJL163', 'ZJL177', 'ZJL51', 'ZJL149', 'ZJL169', 'ZJL28', 'ZJL143', 'ZJL156', 'ZJL162', 'ZJL100', 'ZJL171', 'ZJL180', 'ZJL29', 'ZJL187', 'ZJL119', 'ZJL128', 'ZJL10', 'ZJL69', 'ZJL150', 'ZJL7', 'ZJL165', 'ZJL135', 'ZJL145', 'ZJL19', 'ZJL38', 'ZJL104', 'ZJL46', 'ZJL96', 'ZJL53', 'ZJL153', 'ZJL95', 'ZJL115', 'ZJL93', 'ZJL140', 'ZJL174', 'ZJL83', 'ZJL87', 'ZJL102', 'ZJL54', 'ZJL199', 'ZJL114', 'ZJL15', 'ZJL111', 'ZJL178', 'ZJL58', 'ZJL131', 'ZJL137', 'ZJL31', 'ZJL61', 'ZJL144', 'ZJL59', 'ZJL32', 'ZJL152', 'ZJL37', 'ZJL30', 'ZJL139', 'ZJL21', 'ZJL197', 'ZJL1', 'ZJL166', 'ZJL45', 'ZJL194', 'ZJL168', 'ZJL71', 'ZJL18', 'ZJL172', 'ZJL184', 'ZJL92', 'ZJL170', 'ZJL105', 'ZJL8', 'ZJL173', 'ZJL80', 'ZJL109', 'ZJL147', 'ZJL75', 'ZJL14', 'ZJL183', 'ZJL40', 'ZJL129', 'ZJL99', 'ZJL141', 'ZJL161', 'ZJL192', 'ZJL108', 'ZJL76', 'ZJL182', 'ZJL3', 'ZJL90', 'ZJL48', 'ZJL91', 'ZJL86', 'ZJL64', 'ZJL188', 'ZJL157', 'ZJL65', 'ZJL122', 'ZJL88', 'ZJL181', 'ZJL26', 'ZJL62', 'ZJL35', 'ZJL123', 'ZJL70', 'ZJL132', 'ZJL113', 'ZJL57', 'ZJL196', 'ZJL41', 'ZJL121', 'ZJL5', 'ZJL9', 'ZJL4', 'ZJL189', 'ZJL118', 'ZJL85', 'ZJL106', 'ZJL82', 'ZJL55', 'ZJL68', 'ZJL49'] 
elif args.name == "mobilenet":
    model_dir = "../models/mobilenet/"
    model_name = "mobile"
elif args.name == "resnet50":
    model_dir = "../models/resnet50/"
    model_name = "resnet50"

model = Model(inputs=base_model.input, outputs= predictions)
model.load_weights(model_dir + model_name + '_baseline_model.h5')
"""
base_model = MobileNet(include_top=True, weights=None,
                           input_tensor=None, input_shape=None,
                           pooling=None, classes=40)
base_model.load_weights("/home/anhaoran/codes/zero-shot-cnn/Baselines/zero_shot_learning_baseline/model/mobile_Animals_wgt.h5")
model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('global_average_pooling2d_1').output)
                  """
model.summary()

#Predicting
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)
    x = np.array(x)
    x = np.exp(x)
    x.astype('float32')
    if x.ndim == 1:
        sumcol = sum(x)
        for i in range(x.size):
            x[i] = x[i]/float(sumcol)
    if x.ndim > 1:
        sumcol = x.sum(axis = 0)
        for row in x:
            for i in range(row.size):
                row[i] = row[i]/float(sumcol[i])
    return x
#data_generator = ImageDataGenerator( rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True )
#data_generator.fit(data_x)
predictions = model.predict(data_x * 1.0 / 255)
predictions_all = list()
images_learned_all = list()
nine5_sum = 0
for i in range(len(predictions)):
    predictions[i] = softmax(predictions[i])
    #print(predictions[i])
    max_prob = np.max(predictions[i])
    if max_prob >= 0.75:
        label = np.argmax(max_prob)
        #print(max_prob)
        predictions_all.append(classes[label])
        images_learned_all.append(images_all[i])
        nine5_sum += 1
print(nine5_sum, "in ", len(predictions), "--->", nine5_sum * 1.0 / len(predictions))
images_all = images_learned_all
data_all = {'predictions_all':predictions_all, 
            'images_all':images_all}
savename = "../results/" + model_name + '_predictions_all.pickle'
fsave = open(savename, 'wb')
pickle.dump(data_all, fsave)
fsave.close()
