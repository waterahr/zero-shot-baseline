# -*- coding: utf-8 -*-
import numpy as np
import cv2
from sklearn.decomposition import PCA
import pickle

pca = PCA(n_components=1280)

sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.SURF()
 
"""
#Loading the dataset: train_x(each row as a image array) and train_y(attributes per image)
data_train = open(r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/train.txt')
data_train = data_train.readlines()
print(data_train[0])
path = r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/train/'
length = len(data_train)
features = []
#i = 0
for i in range(length):
    m,n = data_train[i].split()
    #img = image.load_img(path + m)
    img = cv2.imread(path+m)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)
    #print(np.shape(kp))
    #print(type(kp),type(kp[0]))
    #Keypoint数据类型分析 http://www.cnblogs.com/cj695/p/4041399.html
    #print(kp[0].pt)
    #计算每个点的sift
    des = sift.compute(gray, kp)
    #print(type(kp),type(des))
    #des[0]为关键点的list，des[1]为特征向量的矩阵
    #print(type(des[0]), type(des[1]))
    #print(des[0],des[1])
    #可以看出共有885个sift特征，每个特征为128维
    #print(des[1].shape)
    feature = des[1].reshape((-1,))
"""

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

images_all = list()
labels_all = list()
path_test = r'/home/anhaoran/data/zero-shot-tianchi/dataset_B/DatasetB_20180919/test/'
path_train = r'/home/anhaoran/data/zero-shot-tianchi/dataset_B/DatasetB_20180919/train/'
length = len(data_dict)
idx = 0
features_all = []
for img_name, img_label in data_dict.items():
    if img_label != "test":
        img = cv2.imread(path_train + img_name)
    else:
        img = cv2.imread(path_test + img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)
    des = sift.compute(gray, kp)
    images_all.append(img_name)
    labels_all.append(img_label)
    feature = np.reshape(des[1], (-1, ))
    #print("***", np.shape(feature))
    if feature.shape[0] < 1280:
        len_front = (1280 - feature.shape[0]) // 2
        len_tail = (1280 - feature.shape[0]) - len_front
        feature = np.pad(feature, (len_front, len_tail), 'constant')
    else:
        #feature = np.reshape(des[1], (1, -1))
        #print("***", feature.shape)
        #feature = pca.fit_transform(feature)
        #feature = feature.reshape((-1,))
        feature = feature[:1280]
    print(np.shape(feature))
    features_all.append(feature)
    
data_all = {'features_all':np.array(features_all, dtype="float64"), 
            'labels_all':labels_all,
            'images_all':images_all}
savename = "../results/" + "sift" + '_features_all.pickle'
fsave = open(savename, 'wb')
pickle.dump(data_all, fsave)
fsave.close()

    
