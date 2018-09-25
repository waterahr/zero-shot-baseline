# zero-shot-learning-keras_application_models

### Description
2018之江杯全球人工智能大赛-零样本图像目标识别

### Competation
零样本学习是AI识别方法之一。简单来说就是识别从未见过的数据类别，即训练的分类器不仅仅能够识别出训练集中已有的数据类别，还可以对于来自未见过的类别的数据进行区分。这是一个很有用的功能，使得计算机能够具有知识迁移的能力，并无需任何训练数据，很符合现实生活中海量类别的存在形式。本次比赛要求参赛选手提交对测试样本的类别预测值，除官方数据外，不可使用任何外部图片数据进行训练及预训练的模型。主办方提供一个图片数据集，按照类别4：1划分为训练集和测试集，本次比赛训练时禁止使用测试集数据，我们提供一个预训练的类别词向量 和标记的类别属性供选手使用，同时选手可基于外部语料自行训练类别词向量，使用类别的外部关联属性知识库等数据进行辅助训练。本次竞赛主办方有权要求参赛者提交源代码供审查。审查不通过者，取消名次。


### Dataset
初赛数据集分Dataset A和Dataset B两部分, 复赛数据集分为Dataset C和Dataset D两部分。训练数据见文件train，测试数据见文件test。

标签见train.txt，文件结构如下：
a6394b0f513290f4651cc46792e5ac86.jpeg     ZJL1
每张图片一行，每个字段以Tab隔开，分别表示：图片名     标签 ID  。

标签ID与真实类别英文名称对照，见label_list.txt，文件结构如下：
ZJL109     gondola
每个类别一行，每个字段以tab隔开，分别表示：
标签 ID    标签英文名
预训练的类别词向量文件class_wordembeddings.txt(基于Glove预训练的300维词向量仅供参考)

类别级属性标注，见attributes_per_class.txt，文件结构如下：
ZJL109      0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0     0

每张类别一行，每个字段以Tab隔开，分别表示：类别ID     属性标注（顺序同attribute_list.txt，基于众包标注仅供参考）。

属性ID与属性英文名称对照，见attribute_list.txt，文件结构如下：
1     is animal
每个属性一行，每个字段以tab隔开，分别表示：属性 ID属性英文名


### Running
Train the feature extractor (vgg19, densenet121, inceptionv3, mobilenet):
```bash
python baseline.py
```
the models could be saved as follows:
models/
├── densenet121
│   └── DenseNet121_baseline_model.h5
├── inceptionv3
│   ├── inceptionv3_2018-08-24_baseline_model.h5
│   ├── inceptionv3_2018-08-24_epoch01_10.35.hdf5
│   ├── inceptionv3_2018-08-24_epoch09_10.26.hdf5
│   ├── inceptionv3_2018-08-24_epoch100_5.48.hdf5
│   ├── inceptionv3_2018-08-24_epoch20_9.74.hdf5
│   ├── inceptionv3_2018-08-24_epoch25_9.72.hdf5
│   ├── inceptionv3_2018-08-25_baseline_model.h5
│   ├── inceptionv3_2018-08-25_epoch100_0.86.hdf5
│   ├── inceptionv3_2018-08-25_epoch25_3.20.hdf5
│   ├── inceptionv3_2018-08-25_epoch50_1.92.hdf5
│   ├── inceptionv3_2018-08-25_epoch75_1.25.hdf5
│   ├── inceptionv3_2018-08-26_epoch25_7.91.hdf5
│   ├── inceptionv3_2018-08-26_epoch50_7.30.hdf5
│   ├── inceptionv3_2018-08-28_baseline_model.h5
│   ├── inceptionv3_2018-08-28_epoch100_0.22.hdf5
│   ├── inceptionv3_2018-08-28_epoch160_0.20.hdf5
│   ├── inceptionv3_2018-08-28_epoch180_0.10.hdf5
│   ├── inceptionv3_2018-08-28_epoch20_3.52.hdf5
│   ├── inceptionv3_2018-08-28_epoch40_0.95.hdf5
│   ├── inceptionv3_2018-08-28_epoch60_0.53.hdf5
│   ├── inceptionv3_2018-08-28_epoch80_0.33.hdf5
│   ├── inceptionv3_epoch25_valloss35.67.hdf5
│   ├── logs
│   │   ├── log_2018-08-22.csv
│   │   ├── log_2018-08-23.csv
│   │   ├── log_2018-08-24.csv
│   │   ├── log_2018-08-25.csv
│   │   ├── log_2018-08-26.csv
│   │   └── log.csv
│   ├── nohup_2018-08-22.out
│   ├── nohup_2018-08-23.out
│   ├── nohup_2018-08-24.out
│   ├── nohup_2018-08-25.out
│   ├── nohup_2018-08-26.out
│   └── nohup_2018-08-28.out
├── mobilenet
│   └── mobile_baseline_model.h5
├── resnet50
│   └── logs
└── vgg19
    ├── logs
    │   └── log.csv
    ├── vgg19_2018-08-27_baseline_model.h5
    └── vgg19——2018-08-27_epoch180_8.45.hdf5

Extract image features:
```bash
python feature_extract.py 
```
the features could be saved as follows:
results/
├── DenseNet121_features_all.pickle
├── inceptionv3_2018-08-28_features_all.pickle
├── mobile_features_all.pickle
├── submission_2018-08-25.txt
├── submission_2018-08-28_mobile.txt
├── submission_2018-08-29_00:17:34_inceptionv3_2018-08-28.txt
└── submission_2018-08-29_01:13:59_Dense121.txt

Implement zero-shot learning:
```bash
python MDP.py 
```
the submission.txt could be saved in ./results/.

