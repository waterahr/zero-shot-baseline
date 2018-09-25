from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn.linear_model as models
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
import pandas as pd
import sys
import datetime
import os
import argparse
import pickle
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def parse_arg():
    parser = argparse.ArgumentParser(description='training of the baseline...')
    parser.add_argument('--gpus', type=str, default='',
                        help='gpu device\'s ID need to be used')
    parser.add_argument('--features', type=str, default='',
                       help='the features file has been extracted')
    parser.add_argument('--predictions', type=str, default='',
                       help='the predictions file has been done')
    parser.add_argument('--weights', type=str, default='',
                       help='the weights file has been done')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

args = parse_arg()
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

# Load image features & predictions
file_feature = args.features
file_prediction = args.predictions
fdata = open(file_feature, 'rb')
features_dict = pickle.load(fdata)  # variables come out in the order you put them in
fdata.close()
features_all = features_dict['features_all']
labels_all = features_dict['labels_all']
images_all = features_dict['images_all']
print("The shape of features_all is : ", features_all.shape)
print("The shape of labels_all is : ", np.shape(labels_all))
print("The shape of images_all is : ", np.shape(images_all))
images_learned_all = dict()
predictions_all = dict()
if file_prediction != '':
    fdata = open(file_prediction, 'rb')
    predictions_dict = pickle.load(fdata)  # variables come out in the order you put them in
    fdata.close()
    predictions_all = predictions_dict['predictions_all']
    images_learned_all = predictions_dict['images_all']
print("The shape of predictions_all is : ", np.shape(predictions_all))
print("The shape of images_learned_all is : ", np.shape(images_learned_all))
#print(features_dict)


# Calculate number
train_num = 0 
test_num = 0
for lab in labels_all:
    if lab[0] != 't':
        train_num += 1
    else:
        test_num += 1
print("The training samples : ", train_num, '\t', "The testing samples : ", test_num)


#Loading attributes per class
f = open(r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/attributes_per_class.txt')
lines_attr = f.readlines()
attributes = dict()
for each in lines_attr:
    tokens = each.split()
    #print(tokens)
    label = tokens[0]
    attr_list = list()
    for idx in range(1, 26):
        attr_list.append(float(tokens[idx]))
    if not (len(attr_list) == 25):
        print(('attributes number error\n'))
        exit()
    attributes[label] = attr_list
#print(attributes)

list_train = list()
list_test = list()
for img_lab in attributes.keys():
    if len(img_lab[3:]) >= 3 and img_lab[3:] > "200":
        list_test.append(img_lab)
    else:
        list_train.append(img_lab)
print("The length of list_train is : ", len(list_train))
#print(list_train)
print("The length of list_test is : ", len(list_test))
#print(list_test)


# Calculate prototypes (cluster centers)
print(np.max(abs(features_all)))
features_all = features_all/np.max(abs(features_all))
dim_f = features_all.shape[1]
prototypes_train = np.ndarray((len(list_train), dim_f))

dim_a = 25
attributes_train = np.ndarray((len(list_train), dim_a))
attributes_test = np.ndarray((len(list_test), dim_a))

for i in range(len(list_train)):
    label = list_train[i]
    idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
    prototypes_train[i, :] = np.mean(features_all[idx, :], axis=0)
    attributes_train[i, :] = np.asarray(attributes[label])

for i in range(len(list_test)):
    label = list_test[i]
    attributes_test[i, :] = np.asarray(attributes[label])

print("The prototypes_train: ", prototypes_train.shape)
print(prototypes_train)
print("The attributes_train: ", attributes_train.shape)
print(attributes_train)
print("The attributes_test: ", attributes_test.shape)
print(attributes_test)


# Structure learning
"""
LASSO = models.Lasso(alpha=0.01)
LASSO.fit(attributes_train.transpose(), attributes_test.transpose())
W = LASSO.coef_

# Image prototype synthesis
prototypes_test = (np.dot(prototypes_train.transpose(), W.transpose())).transpose()
"""
model = Sequential()#第一层<br>#Dense就是全连接层
model.add(Dense(512, input_shape=(190, ))) #输入维度, 512==输出维度
model.add(Activation('relu')) #激活函数
model.add(Dropout(0.5)) #dropout<br><br>#第二层
model.add(Dense(40))
model.add(Activation('relu')) #激活函数
#损失函数设置、优化函数，衡量标准
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
#训练，交叉验证
if args.weights == '':
    model.fit(attributes_train.transpose(), attributes_test.transpose(), epochs=1000, batch_size=128, verbose=1, validation_split=0.1)
    model.save("mdp.h5")
else:
    model.load_weights(args.weights)

prototypes_test = model.predict(prototypes_train.transpose()).transpose()

# Prediction
label = 'test'
idx = [pos for pos, lab in enumerate(labels_all) if lab == label]
features_test = features_all[idx, :]
images_test = [images_all[i] for i in idx]
prediction = list()

#model = load_model(args.model)
#path_test = r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/test/'
for i in range(len(idx)):
    #img = image.load_img(path_test + img_name, target_size=(args.image_size, args.image_size, 3))
    #data_x = image.img_to_array(img)
    #data_x = np.expand_dims(data_x, axis=0)
    temp = np.repeat(np.reshape((features_test[i, :]), (1, dim_f)), len(list_test), axis=0)
    distance = np.sum((temp - prototypes_test)**2, axis=1)
    pos = np.argmin(distance)
    prediction.append(list_test[pos])
    

now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
start_findex = file_feature.index("results") + 8
end_findex = file_feature.index("_features")
start_pindex = file_prediction.index("results") + 8
end_pindex = file_prediction.index("_predictions")
txtName = "../results/submission_" + now_time + '_' + file_feature[start_findex:end_findex] + '+' + file_prediction[start_pindex:end_pindex] + ".txt"
f=open(txtName, "w")
for i in range(len(prediction)):
    img_name = images_all[train_num + i]
    if img_name not in images_learned_all:
        new_context = img_name +'\t' + prediction[i] + '\n'
    else:
        new_context = img_name +'\t' + predictions_all[images_learned_all.index(img_name)] + '\n'
    f.write(new_context)
f.close()
print(txtName, " has benn saved!")