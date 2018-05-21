### pytorch-yolo2
Convert https://pjreddie.com/darknet/yolo/ into pytorch. This repository is trying to achieve the following goals.
- [x] implement RegionLoss, MaxPoolStride1, Reorg, GolbalAvgPool2d
- [x] implement route layer
- [x] detect, partial, valid functions
- [x] load darknet cfg
- [x] load darknet saved weights
- [x] save as darknet weights
- [x] fast evaluation
- [x] pascal voc validation
- [x] train pascal voc
- [x] LMDB data set
- [x] Data augmentation
- [x] load/save caffe prototxt and weights
- [x] **reproduce darknet's training results**
- [x] [convert weight/cfg between pytorch caffe and darknet](https://github.com/marvis/pytorch-caffe-darknet-convert)
- [x] add focal loss

---
#### Detection Using A Pre-Trained Model
```
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg yolo.weights data/dog.jpg
```
You will see some output like this:
```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32
    1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32
    ......
   30 conv    425  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 425
   31 detection
Loading weights from yolo.weights... Done!
data/dog.jpg: Predicted in 0.014079 seconds.
truck: 0.934711
bicycle: 0.998013
dog: 0.990524
```
---
#### Real-Time Detection on a Webcam
```
python demo.py cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights
```
---

#### Training YOLO on VOC
##### Get The Pascal VOC Data
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
##### Generate Labels for VOC
```
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
##### Modify Cfg for Pascal Data
Change the cfg/voc.data config file
```
train  = train.txt
valid  = 2007_test.txt
names = data/voc.names
backup = backup
```
##### Download Pretrained Convolutional Weights
Download weights from the convolutional layers
```
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```
or run the following command:
```
python partial.py cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23
```
##### Train The Model
```
python train.py cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
##### Evaluate The Model
```
python valid.py cfg/voc.data cfg/yolo-voc.cfg yolo-voc.weights
python scripts/voc_eval.py results/comp4_det_test_
```
mAP test on released models
```
yolo-voc.weights 544 0.7682 (paper: 78.6)
yolo-voc.weights 416 0.7513 (paper: 76.8)
tiny-yolo-voc.weights 416 0.5410 (paper: 57.1)

```
---
#### Focal Loss 
A implementation of paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

We get the results by using Focal Loss to replace CrossEntropyLoss in RegionLosss.

 gama       | training set | val set | mAP@416 | mAP@544 | Notes
---         |---           |---      |---      |---      |---
 0          |VOC2007+2012  | VOC2007 | 73.05   |  74.69  | std-Cross Entropy Loss
 1          |VOC2007+2012  | VOC2007 | 73.63   |  75.26  | Focal Loss
 2          |VOC2007+2012  | VOC2007 |**74.08**|**75.49**| Focal Loss
 3          |VOC2007+2012  | VOC2007 |  73.73  |  75.20  | Focal Loss
 4          |VOC2007+2012  | VOC2007 |  73.53  |  74.95  | Focal Loss


---
#### Problems
##### 1. Running variance difference between darknet and pytorch
Change the code in normalize_cpu to make the same result
```
normalize_cpu:
x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));
``` 
#### Training on your own data
1. Padding your images into square size and produce the corresponding label files.
2. Modify the resize strageties in listDataset. Currently the resize scales range from 320 ~ 608, and the resize intervals is 64, which should be equal to batch_size or several times of batch_size. 
3. Add warm up learning rate (scales=**0.1**,10,.1,.1)
4. Train your model as VOC does.

---
#### License
MIT License (see LICENSE file).

#### Contribution
Thanks for the contributions from @iceflame89 for the image augmentation and @huaijin-chen for focal loss.
