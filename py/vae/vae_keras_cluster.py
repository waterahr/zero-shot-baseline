#! -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Model
from keras import backend as K
import imageio
#from keras.datasets import mnist
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
# from keras.datasets import fashion_mnist as mnist
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import argparse
from keras.callbacks import ModelCheckpoint

monitor = 'loss'
checkpointer = ModelCheckpoint(filepath = 'vae_epoch{epoch:02d}_valloss{'+ monitor + ':.2f}.hdf5',
                               monitor = monitor,
                               verbose=1, 
                               save_best_only=True, 
                               save_weights_only=False,
                               mode='auto', 
                               period=25)

def parse_arg():
    parser = argparse.ArgumentParser(description='training of the baseline...')
    parser.add_argument('-l', '--latent_dim', type=int, default=10,#380
                        help='')
    parser.add_argument('-f', '--filters', type=int, default=32,#16
                        help='')
    parser.add_argument('-i', '--intermediate_dim', type=int, default=256,
                        help='')
    args = parser.parse_args()
    return args

args = parse_arg()


batch_size = 32
epochs = 500
img_dim = 64
num_classes = 190
#super parameters
latent_dim = args.latent_dim#380,2,5,10#trained:100,380,10,
filters = args.filters#32,16#trained:16
intermediate_dim = args.intermediate_dim#1024,256#trained:256


"""
# 加载MNIST数据集
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, img_dim, img_dim, 1))
x_test = x_test.reshape((-1, img_dim, img_dim, 1))
"""
#"""
#Loading the dataset: train_x(each row as a image array) and train_y(attributes per image)
data_atten = pd.read_csv(r'../../test.csv')
data_atten = data_atten.set_index('0')
data_train = open(r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/train.txt')
data_train = data_train.readlines()
print(data_train[0])
path = r'/home/anhaoran/data/zero-shot-tianchi/dataset_A/train/train/'
length = len(data_train)
train_x = np.zeros((length, img_dim, img_dim, 3))
train_y = np.zeros((length, ), dtype="int64")#num_classes
labels = []
for i in range(length):
    m,n = data_train[i].split()
    if not n in labels:
        labels.append(n)
    #img = image.load_img(path + m)
    img = image.load_img(path + m, target_size=(img_dim, img_dim, 3))
    train_x[i] = image.img_to_array(img) / 255.
    #train_y[i][labels.index(n)] = 1
    train_y[i] = labels.index(n)
x_train, x_test, y_train_, y_test_ = train_test_split(train_x,train_y, test_size=0.2, random_state=0)
print("The shape of the X_train is: ", x_train.shape)
print("The shape of the y_train is: ", y_train_.shape)
#"""
"""
trainDataDir = "/home/anhaoran/data/zero-shot-tianchi/dataset_Ac/train/"
valDataDir = "/home/anhaoran/data/zero-shot-tianchi/dataset_Ac/val/"
classes_train = os.listdir(trainDataDir)
classes_val = os.listdir(valDataDir)
print("The total number of the classes in the dataset is :", len(classes_train), ' for train, ', len(classes_val), ' for validation')
print(classes_train, classes_val)
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
train_generator =  datagen.flow_from_directory( trainDataDir, target_size = (img_dim, img_dim), batch_size = batch_size, classes = classes_train )
val_generator = datagen.flow_from_directory( valDataDir, target_size = (img_dim, img_dim), batch_size = batch_size, classes = classes_val)
"""



# 搭建模型
x = Input(shape=(img_dim, img_dim, 3))
h = x

for i in range(2):
    filters *= 2
    h = Conv2D(filters=filters,
               kernel_size=3,
               strides=2,
               padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(filters=filters,
               kernel_size=3,
               strides=1,
               padding='same')(h)
    h = LeakyReLU(0.2)(h)


h_shape = K.int_shape(h)[1:]
h = Flatten()(h)
z_mean = Dense(latent_dim)(h) # p(z|x)的均值
z_log_var = Dense(latent_dim)(h) # p(z|x)的方差

encoder = Model(x, z_mean) # 通常认为z_mean就是所需的隐变量编码


z = Input(shape=(latent_dim,))
h = z
h = Dense(np.prod(h_shape))(h)
h = Reshape(h_shape)(h)

for i in range(2):
    h = Conv2DTranspose(filters=filters,
                        kernel_size=3,
                        strides=1,
                        padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(filters=filters,
                        kernel_size=3,
                        strides=2,
                        padding='same')(h)
    h = LeakyReLU(0.2)(h)
    filters //= 2

x_recon = Conv2DTranspose(filters=3,
                          kernel_size=3,
                          activation='sigmoid',
                          padding='same')(h)


decoder = Model(z, x_recon) # 解码器
generator = decoder


z = Input(shape=(latent_dim,))
y = Dense(intermediate_dim, activation='relu')(z)
y = Dense(num_classes, activation='softmax')(y)

classfier = Model(z, y) # 隐变量分类器


# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
x_recon = decoder(z)
y = classfier(z)


class Gaussian(Layer):
    """这是个简单的层，只为定义q(z|y)中的均值参数，每个类别配一个均值。
    输出也只是把这些均值输出，为后面计算loss准备，本身没有任何运算。
    """
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z * 0 + K.expand_dims(self.mean, 0)
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])

gaussian = Gaussian(num_classes)
z_prior_mean = gaussian(z)


# 建立模型
vae = Model(x, [x_recon, z_prior_mean, y])

# 下面一大通都是为了定义loss
z_mean = K.expand_dims(z_mean, 1)
z_log_var = K.expand_dims(z_log_var, 1)

lamb = 5 # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
xent_loss = 0.5 * K.mean((x - x_recon)**2, 0)
kl_loss = - 0.5 * (1 + z_log_var - K.square(z_mean - z_prior_mean) - K.exp(z_log_var))
kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
vae_loss = lamb * K.sum(xent_loss) + K.sum(kl_loss) + K.sum(cat_loss)


vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()


#"""
vae.fit(x_train, 
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        callbacks = [checkpointer])
vae.save_weights("vae_latentDim"+str(latent_dim)+"_intermediateDim"+str(intermediate_dim)+"_filters"+str(filters)+".h5")
#"""
#vae.load_weights("vae_latentDim"+str(latent_dim)+"_intermediateDim"+str(intermediate_dim)+"_filters"+str(filters)+".h5")


means = K.eval(gaussian.mean)
x_train_encoded = encoder.predict(x_train)
y_train_pred = classfier.predict(x_train_encoded).argmax(axis=1)
#print(y_train_pred)
#print(y_train_pred.shape)
#print(y_train_pred.min())
x_test_encoded = encoder.predict(x_test)
y_test_pred = classfier.predict(x_test_encoded).argmax(axis=1)


def cluster_sample(path, category=0):
    """观察被模型聚为同一类的样本
    """
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    idxs = np.where(y_train_pred == category)[0]
    for i in range(n):
        for j in range(n):
            digit = x_train[np.random.choice(idxs)]
            digit = digit.reshape((img_dim, img_dim, 3))
            figure[i * img_dim: (i + 1) * img_dim,
            j * img_dim: (j + 1) * img_dim] = digit
    imageio.imwrite(path, figure * 255)


def random_sample(path, category=0, std=1):
    """按照聚类结果进行条件随机生成
    """
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            noise_shape = (1, latent_dim)
            z_sample = np.array(np.random.randn(*noise_shape)) * std + means[category]
            x_recon = generator.predict(z_sample)
            digit = x_recon[0].reshape((img_dim, img_dim, 3))
            figure[i * img_dim: (i + 1) * img_dim,
            j * img_dim: (j + 1) * img_dim] = digit
    imageio.imwrite(path, figure * 255)


samples_dir = "samples_latentDim"+str(latent_dim)+"_intermediateDim"+str(intermediate_dim)+"_filters"+str(filters)
if not os.path.exists(samples_dir):
    os.mkdir(samples_dir)

for i in range(10):
    cluster_sample((samples_dir+'/x_train_samples_category_%s.png') % i, i)
    random_sample((samples_dir+'/x_recon_samples_category_%s.png') % i, i)
    


right = 0.
for i in range(10):
    _ = np.bincount(y_train_[y_train_pred == i])
    right += _.max()

print('train acc: %s' % (right / len(y_train_)))


right = 0.
for i in range(10):
    _ = np.bincount(y_test_[y_test_pred == i])
    right += _.max()

print('test acc: %s' % (right / len(y_test_)))

print(samples_dir)