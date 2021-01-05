"""
This script tests the rotation equivariance of a trained network on rot-MNIST digits
"""

import argparse
import os
import random
import sys
import time
#import urllib2
from urllib.request import urlopen
import zipfile
sys.path.append('../')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mnistmodel import DeepMNIST

from skimage import io

def settings(args):
   #print('Running settings fn')
   
   # Other options
   if args.default_settings:
      args.n_epochs = 2000
      args.batch_size = 72
      args.learning_rate = 0.05#0.0076
      args.std_mult = 0.7
      args.delay = 12
      args.phase_preconditioner = 7.8
      args.filter_gain = 2
      args.filter_size = 5
      args.n_rings = 4
      args.n_filters = 8
      args.is_classification = True
      args.dim = 28
      args.crop_shape = 0
      args.n_channels = 1
      args.n_classes = 10
      args.lr_div = 10.
      args.model_path = './models/model_3235.pth'
      args.dataset = './data/mnist_rotation_new/train_100/30.npz'
      args.featmap_save_location = './data/mnist_rotation_new/train_100/featmaps/'

   args.log_path = add_folder('./logs')
   args.checkpoint_path = add_folder('./checkpoints') + '/model.ckpt'
   return args

def add_folder(folder_name):
   if not os.path.exists(folder_name):
      os.mkdir(folder_name)
      print('Created {:s}'.format(folder_name))
   return folder_name


def get_learning_rate(args, current, best, counter, learning_rate):
   """If have not seen accuracy improvement in delay epochs, then divide 
   learning rate by 10
   """
   if current > best:
      best = current
      counter = 0
   elif counter > args.delay:
      learning_rate = learning_rate / args.lr_div
      counter = 0
   else:
      counter += 1
   return (best, counter, learning_rate)

def save_featmaps(featmaps, save_path, save_idx):

   for itr in range(featmaps.shape[0]):
      featmaps = featmaps.reshape((featmaps.shape[0], featmaps.shape[1], featmaps.shape[2], -1))
      print(featmaps.shape)
      n_channels = featmaps.shape[3]
      #saving all the channels as images
      for channel in range(n_channels):
         featmap = featmaps[itr,:,:,channel]
         featmap = featmap * 255.0
         io.imsave(save_path+str(itr) + '_'+ str(channel)+'.jpg', featmap.astype(np.uint8))

def main(args):

   ##### SETUP AND LOAD DATA #####
   args = settings(args)

   # choosing the device to run the model
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(device)


   model = DeepMNIST(args).to(device)
   model.load_state_dict(torch.load(args.model_path))
   
   # load a batch of rotations
   img_set = np.load(args.dataset)
   img_set = img_set[img_set.files[0]]
   print(img_set.shape)
   print(img_set.max())

   img_set = torch.cuda.FloatTensor(img_set) if device=="cuda" else torch.FloatTensor(img_set)
   images = img_set.to(device)
   print(images.shape)


   logits, featmap = model(images)
   featmap = featmap.cpu().numpy()
   predictions = torch.argmax(logits, dim=1).cpu().detach().numpy()
   print(predictions.shape)
   print(featmap.shape)

   # saving the featmap for every image with idx (i,j,k)
   save_idx = [0, 0, 0]

   save_featmaps(featmap, args.featmap_save_location, save_idx)





if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--data_dir", help="data directory", default='./data')
   parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
   parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool, default=False)
   main(parser.parse_args())


     