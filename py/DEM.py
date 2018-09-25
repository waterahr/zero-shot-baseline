from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn.linear_model as models
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import sys
import datetime
import os
import argparse
import pickle
import scipy.io as sio
import random
import h5py
from numpy import *
import kNN
from tqdm import tqdm
import re
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def parse_arg():
    parser = argparse.ArgumentParser(description='training of the baseline...')
    parser.add_argument('--gpus', type=str, default='',
                        help='gpu device\'s ID need to be used')
    parser.add_argument('--features', type=str, default='',
                       help='the features file has been extracted')
    parser.add_argument('--attributes', type=str, default='',
                       help='the attributes file has been predicted using the model')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='the batch size to train')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# # data shuffle
def data_iterator():
    """ A simple data iterator """
    global batch_size
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(x))
        np.random.shuffle(idxs)
        shuf_visual = x[idxs]
        shuf_att = att[idxs]
        for batch_idx in range(0, len(x), batch_size):
            visual_batch = shuf_visual[batch_idx:batch_idx+batch_size]
            visual_batch = visual_batch.astype("float32")
            att_batch = shuf_att[batch_idx:batch_idx+batch_size]
            yield att_batch, visual_batch

args = parse_arg()
batch_size = args.batch_size
dim_att = 25
dim_fea = 1024
nb_epoch = 1000000
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

# Load image features
file_feature = args.features
fdata = open(file_feature, 'rb')
features_dict = pickle.load(fdata)  # variables come out in the order you put them in
fdata.close()
features_all = features_dict['features_all']
labels_all = features_dict['labels_all']
images_all = features_dict['images_all']
print("The shape of features_all is : ", features_all.shape)
print("The shape of labels_all is : ", np.shape(labels_all))
print("The shape of images_all is : ", np.shape(images_all))
#print(features_dict)

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

# Calculate number
train_num = 0 
test_num = 0
for lab in labels_all:
    if lab[0] != 't':
        train_num += 1
    else:
        test_num += 1
print("The training samples : ", train_num, '\t', "The testing samples : ", test_num)

#f=h5py.File('./data/AwA_data/attribute/Z_s_con.mat','r')
#Another solution: Using the predictions of the attributes-detection_model
data_atten = pd.read_csv(r'../test.csv')
data_atten = data_atten.set_index('0')
att = np.zeros((train_num, dim_att))
for i in range(train_num):
    n = labels_all[i]
    att[i] = data_atten.loc[n]
print("The shape of train samples' attributes is : ", att.shape)

#f=sio.loadmat('./data/AwA_data/train_googlenet_bn.mat')
x = features_all[:train_num, :]
print("The shape of train samples' features is : ", x.shape)

#f=sio.loadmat('./data/AwA_data/test_googlenet_bn.mat')
x_test = features_all[train_num:, :]
print("The shape of test samples' features is : ", x_test.shape)

# # Placeholder
# define placeholder for inputs to network
att_features = tf.placeholder(tf.float32, [None, dim_att])
visual_features = tf.placeholder(tf.float32, [None, dim_fea])

# # Network
# AwA 85 300 1024 ReLu, 1e-2 * regularisers, 64 batch, 0.0001 Adam
W_left_a1 = weight_variable([dim_att, 300])
b_left_a1 = bias_variable([300])
left_a1 = tf.nn.relu(tf.matmul(att_features, W_left_a1) + b_left_a1)

W_left_a2 = weight_variable([300, dim_fea])
b_left_a2 = bias_variable([dim_fea])
left_a2 = tf.nn.relu(tf.matmul(left_a1, W_left_a2) + b_left_a2)

# # loss
loss_a = tf.reduce_mean(tf.square(left_a2 - visual_features))    

# L2 regularisation for the fully connected parameters.
regularisers_a = (tf.nn.l2_loss(W_left_a1) + tf.nn.l2_loss(b_left_a1)
                + tf.nn.l2_loss(W_left_a2) + tf.nn.l2_loss(b_left_a2))

# Add the regularisation term to the loss.            
loss_a += 1e-2 * regularisers_a

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_a)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# # Run
iter_ = data_iterator()
for i in tqdm(range(nb_epoch)):
    att_batch_val, visual_batch_val = next(iter_)
    sess.run(train_step, feed_dict={att_features: att_batch_val, visual_features: visual_batch_val})
    
# # Prediction
visual_features_test_prototypes = sess.run(left_a2, feed_dict={att_features: np.asarray(tuple(attributes.values()))[190:]})
#40 * 1024(dim_fea)
# # For x_test, each find the closest in the visual_features_test_prototypes
prediction = []
for i in range(len(x_test)):
    temp = np.repeat(np.reshape((x_test[i, :]), (1, dim_fea)), len(visual_features_test_prototypes), axis=0)
    distance = np.sum((temp - visual_features_test_prototypes)**2, axis=1)
    pos = np.argmin(distance)
    prediction.append(list_test[pos])


now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
start_findex = file_feature.index("results") + 8
end_findex = file_feature.index("_features")
txtName = "../results/submission_DEM_" + now_time + '_' + file_feature[start_findex:end_findex] + ".txt"
f=open(txtName, "w")
for i in range(len(prediction)):
    img_name = images_all[train_num + i]
    new_context = img_name +'\t' + prediction[i] + '\n'
    f.write(new_context)
f.close()
print(txtName, " has benn saved!")