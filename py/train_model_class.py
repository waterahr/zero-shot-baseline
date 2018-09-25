from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import backend as K
from keras.layers import Input 
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
import os
import datetime
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
image_size = 224
batch_size = 64
#class_num = 160
class_num = 190
nb_epoch = 50

#trainDataDir = "/home/anhaoran/data/zero-shot-tianchi/dataset_A/train_as_class/train/"
#valDataDir = "/home/anhaoran/data/zero-shot-tianchi/dataset_A/train_as_class/val/"
trainDataDir = "/home/anhaoran/data/zero-shot-tianchi/dataset_Ac/train/"
valDataDir = "/home/anhaoran/data/zero-shot-tianchi/dataset_Ac/val/"

classes_train = os.listdir(trainDataDir)
"""
for root, dirs, files in os.walk(trainDataDir): 
    print(dirs)
    if dirs not in classes_train and dirs.startswith("ZJL"):
        classes_train.append(dirs)
classes_train = np.asarray(classes_train)
"""
classes_val = os.listdir(valDataDir)
"""
classes_val = []
for root, dirs, files in os.walk(valDataDir):
    print(dirs)
    if dirs not in classes_val and dirs.startswith("ZJL"):
        classes_val.append(dirs)
classes_val = np.asarray(classes_val)
"""
print("The total number of the classes in the dataset is :", len(classes_train), ' for train, ', len(classes_val), ' for validation')
print(classes_train, classes_val)

#base_model = MobileNet(weights = None, include_top = False, pooling = 'avg' )
#base_model = DenseNet121( weights = None, include_top = False, pooling='avg' )
base_model = InceptionV3( weights = None, include_top = False, pooling = 'avg')
base_modelOutput = base_model.output
#x = Flatten()( base_modelOutput )
x = Dense(1024, activation = 'relu')( base_modelOutput )
predictions = Dense(class_num, activation = 'softmax')( x )
model = Model(inputs = base_model.input, outputs = predictions)
#with tf.device("/cpu:0"):
#    model = Model(inputs = base_model.input, outputs = predictions)
#make the model parallel
model = multi_gpu_model(model, gpus=2)
model.summary()
'''
for layer in base_model.layers:
    layer.trainable = False
'''
model.compile( loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'] )

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



#modelDir = '../models/densenet121/'
modelDir = '../models/inceptionv3/'

if not os.path.exists( modelDir ):
    os.mkdir( modelDir )

#checkpointDir = '../models/densenet121/DenseNet121c160-ep{epoch:03d}-loss{loss:.5f}-acc{acc:.5f}.h5'
checkpointDir = '../models/inceptionv3/InceptionV3c190-ep{epoch:03d}-loss{loss:.5f}-acc{acc:.5f}.h5'
csvlog = CSVLogger(modelDir + 'logs/log.csv')
checkpoint = ModelCheckpoint( checkpointDir, monitor='loss', verbose=1, save_best_only= True, save_weights_only=False, mode='min', period=25)

train_generator =  datagen.flow_from_directory( trainDataDir, target_size = (image_size, image_size), batch_size = batch_size * 2, classes = classes_train )
val_generator = datagen.flow_from_directory( valDataDir, target_size = (image_size, image_size), batch_size = batch_size * 2, classes = classes_val)
#model.fit_generator( trainGenerator, steps_per_epoch = 4375, epochs = 10, verbose = 1, validation_data = valGenerator, validation_steps = 1875 )
model.load_weights(modelDir + 'InceptionV3c190-ep150-loss0.39905-acc0.88435.h5')
model.fit_generator( train_generator, 
              epochs = nb_epoch, 
              verbose = 1,
              validation_data = val_generator,
              callbacks=[checkpoint, csvlog] )
#model.save( modelDir + 'DenseNet121c160_baseline_model.h5')
model.save( modelDir + 'InceptionV3c190_baseline_model.h5')

now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
os.system('mv nohup.out ' + modelDir + 'nohup_' + now_time + '.out')
