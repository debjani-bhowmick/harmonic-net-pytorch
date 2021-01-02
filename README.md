# Classification of Rot-MNIST digits using Harmonic Networks

### Table of Contents

1. [Summary](#summary)
2. [Installation](#installation)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Running Experiments](#experiments)
6. [Results](#results)
7. [Data Description](#data)
8. [References](#references)

### Summary <a name="summary"></a>

This project demonstrates that efficacy of rotation equivariant networks. We use Harmonic Networks [1] to induce rotation equivariance into the deep learning model, and demonstrate on Rotational MNIST (Rot-MNIST) dataset that H-Nets perform better than the regular CNNs in the presence of in-plane rotations. The Pytorch implementation presented here is inspired from the official tensorflow code made publicly available by [1]. For further understanding related to the project as well as the findings, see the blog post [here](https://debjani-ism.medium.com/classification-of-rotational-mnist-digits-using-harmonic-networks-aaabc1406189).

## Installation <a name="installation"></a>

This code has been tested with Anaconda distribution of `Python 3.7.7` and `Pytorch 1.4`. Additional libraries used in the project are: 

cudatoolkit 10.0<br>
numpy 1.19.1<br>
matplotlib 3.2.2<br>
e2CNN 0.1

Beyond performing standard installation of the above packages, no additional installations are required to run code in this project.

## Motivation<a name="motivation"></a>

It is well known that when an object translates within an image, the respective deep features obtained from a trained convolutional neural net (CNN) also translate similarly. Due to this characteristic, regular CNNs are equivariant to translations. Theoretically, this property implies that if the CNN knows to encode the object for one position in the image, it can account for all possible translations without any data augmentation. <b>Do similar properties also hold for other transformations such as rotation and change of scale of the target object?</b>

There are various computer vision problems in real life where CNNs can benefit from rotation equivariance. For example, videos captured from drones contains several instances where the target is rotating in-plane. An example study addressing a similar issue in the context of object tracking showed that the performance of rotation equivariant CNNs is more than 25% better than the regular CNNs [2]. Inspired from this study and several others, this study implements Harmnonic Nets for classification of Rot-MNIST dataset. The implementation has been written from scratch in Pytorch with inspirations forom the Tensorflow code of [1]. Due to some functional differences between two languages, performance might differ.

This study answers the following question:

*Do Harmonic networks perform better than the baseline regular CNN model on the classification task of Rot-MNIST digits?*

<b>Note</b>: This blog post is meant as the final report for my capstone project for Udacity Data Science Nanodegree programme.

## File Descriptions <a name="files"></a>

The project structure as well as the desciption of the contained files is as follows.
* `MNIST_rot/main.py`: main python script for training and testing the rot-equivariant and regular CNN models.
* `MNIST_rot/mnist_model.py`: contains Pytorch model implementations for Harmonic Net and regular CNN.
* `MNIST_rot/visualization_rot-mnist.ipynb`: jupyter notebook for analysis and visualization study of ROT-MNIST dataset.
* `hnet_ops.py`: contains implementaions of different modules needed by the harmoinc network such as convolution, mean pooling, etc.
* `hnet_lite.py`: wrapper/simplified API to convert the H-Net modules to look similar to the standard modules such as Conv2d.

## Data description<a name="data"></a>

The Rot-MNIST dataset can be downloaded from the dropbox [link](https://www.dropbox.com/s/0fxwai3h84dczh0/mnist_rotation_new.zip) made publicly available in the official tensorflow code provided by [1]. Note that in our implementation, script `main.py` automatically downloads the data if not present in the `data` folder, thus explicitly downloading it is not needed. However, in case data analysis is performed with the notebook provided, user needs to first download, unzip and put the data in the `data` folder.

Rot-MNIST contains a total of 62000 images of MNIST digits, each rotated by an arbitrary angle. The full dataset is split into 10000, 2000 and 50000 samples for the training, validation and test sets, respectively. The test and validation sets are chosen such that there are significant slightly out-of-distribution samples in them compared to the training set. Each image is also accompanied by a label from 0-9. Our statistical analysis, also reported in `visualization_rot-mnist.ipynb`, reveals that the data is balanced across the 10 classes with 9-11% of the samples belonging to each of the classes in the training, validation as well as the test sets. For some visuals of the dataset, we recommend the reader to see the `visualization_rot-mnist.ipynb` notebook or visit the blog post [here](https://debjani-ism.medium.com/classification-of-rotational-mnist-digits-using-harmonic-networks-aaabc1406189).

## Experiments <a name="experiments"></a>

For all experiments, which include training and testing of Hamronic Net as well as regular CNN model, the following command needs to be run from within the `MNIST-rot` folder,

>> `python main.py`

We discuss below how the code can be adapted to train or test the models.

### Training the models

For training mode, set `args.train_mode = True` and `args.load_pretrained = False` in `main.py` file. For training Regular CNN model, comment the line `model = RegularCNN(args).to(device)` and comment the line `model = DeepMNIST(args).to(device)`. You are now good to go.

### Testing the models

For test mode, set `args.train_mode = False` and `args.load_pretrained = True` in `main.py` file. For training Regular CNN model, comment the line `model = RegularCNN(args).to(device)` and comment the line `model = DeepMNIST(args).to(device)`.

For test mode, it is assumed that the model has already been trained and saved. Based on whether RegularCNN or RE-CNNN test to be performed, set the path to the pretrained model in the variable `args.pretrained_model`.

## Results<a name="results"></a>

We tested two different models on the Rot-MNIST dataset: Regular CNN and Rotation Equivariant CNN (RE-CNN), built using Harmonic networks. For fair comparison, both models are built with approximately same number of optimization parameters (34K) in their respective networks. For both models, accuracies on validation and test sets were observed to be above 95%. For Regular CNN, the accuracy scores on the validation and test sets are <b>95.81%</b> and <b>95.86%</b>, respectively. For RE-CNN, the scores are <b>96.46%</b> and <b>96.43%</b> on validation and test sets, respectively.  However, in a relative comparison, we see that RE-CNN outperforms Regular CNN for both the subsets of Rot-MNIST data. 

#### Some important findings can be summarized below :<br>
  * Rotation Equivariant networks perform better than regular counterparts for ROT-MNIST dataset.
  * The difference in performance between the two models is small, which implies that the regular CNN can to a large extent handle the in-plane rotations for this dataset. This could be because the 10000 samples from the training set are already sufficient to learn the orientation pattern.
  * It would be of interest to perform the same study on more complex datasets, where we speculate that RE-CNN could provide larger improvements over the baseline.
 
The main findings of the code can also be found at the blog post [here](https://debjani-ism.medium.com/classification-of-rotational-mnist-digits-using-harmonic-networks-aaabc1406189)

## References<a name="references"></a>

[1] D.E. Worrall, S.J. Garbin, D. Turmukhambetov, G.J. Borstow, [Harmonic Networks: Deep Translation and Rotation Equivariance](https://openaccess.thecvf.com/content_cvpr_2017/html/Worrall_Harmonic_Networks_Deep_CVPR_2017_paper.html), CVPR, 2017.

[2] D.K. Gupta, D. Arya, E. Gavves, [Rotation Equivariant Siamese Networks for Tracking
](https://arxiv.org/abs/2012.13078),	arXiv:2012.13078, 2020.
