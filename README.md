# Image-recognition
Image recognition methods from bag of words (BoW), Spatial Pyramid Matching (SPM), Sparse Codeing SPM (ScSPM) to convolutional neural networks (CNN) and CNN-SVM.

## Included methods
-       Bag of words [1]
-       Spatial Pyramid Matching [2]
-       Sparse Coding SPM [3]
-       Convolutional neural networks [4]
-       CNN-SVM [5]

## Requirements
#### Basic (BoW, SPM, ScSPM)
-       Python 2.7
-       NumPy
-       SciPy
-       Scikit-learn
-       OpenCV 3.0.0 + opencv_contrib [installation instructions](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/)  

We use OpenCV here to load the image and extract SIFT descriptor only, you can use any image library if you want.
#### Optional (CNN, CNN-SVM)
-       keras
-       Caffe, pycaffe [installation instructions](http://caffe.berkeleyvision.org/installation.html)  

## Demo
```
$python bow.py --train path/to/trainingData.txt --test path/to/testData.txt
#
             precision    recall  f1-score   support

   airplane       0.75      0.30      0.43        20
 automobile       0.77      0.50      0.61        20
       bird       0.64      0.70      0.67        20
        cat       0.60      0.75      0.67        20
       deer       0.62      0.80      0.70        20
        dog       0.90      0.45      0.60        20
       frog       0.65      0.85      0.74        20
      horse       0.91      1.00      0.95        20
       ship       0.50      0.70      0.58        20

avg / total       0.70      0.67      0.66       180

```

## Usage
An end-to-end script with training and testing is provided.  
**Dataset File** is a text file with **label** and **path** formatting like:

```
#train.txt
label path/to/train1.jpg  
label path/to/train2.jpg  
label path/to/train3.jpg  
```
```
#test.txt
label path/to/test1.jpg
label path/to/test2.jpg
```

Author: CyrusChiu @ntu



-----
[1] CSURKA, Gabriella, et al. Visual categorization with bags of keypoints. In: Workshop on statistical learning in computer vision, ECCV. 2004. p. 1-2.

[2] LAZEBNIK, Svetlana; SCHMID, Cordelia; PONCE, Jean. Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories. In: Computer Vision and Pattern Recognition, 2006 IEEE Computer Society Conference on. IEEE, 2006. p. 2169-2178.

[3] Jianchao Yang, Kai Yu, Yihong Gong, and Thomas Huang. Linear spatial pyramid matching using sparse coding for image classification. CVPR2009

[4] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.

[5] Girshick, Ross, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.

