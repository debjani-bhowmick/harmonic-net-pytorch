'''
This is the main script for training the model deep H-net model 
and evaluating its performance.

Author: Debjani Bhowmick, 2020.
'''

# Importing the necessary dependencies below
import argparse
import os
import random
import sys
import time
from urllib.request import urlopen
import zipfile
sys.path.append('../')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

from mnistmodel import DeepMNIST

class RotMNISTDataset(Dataset):

   def __init__(self, image_set, label_set):
      self.image_set = image_set
      self.label_set = label_set

   def __len__(self):
      return len(self.image_set)

   def __getitem__(self, idx):

      image_sample = torch.from_numpy(self.image_set[idx])
      label_sample = torch.from_numpy(np.asarray(self.label_set[idx]))
      return image_sample, label_sample
      



def download2FileAndExtract(url, folder, fileName):
   print('Downloading rotated MNIST...')
   add_folder(folder)
   zipFileName = folder + fileName
   #request = urllib2.urlopen(url)
   request = urlopen(url)
   with open(zipFileName, "wb") as f :
      f.write(request.read())
   if not zipfile.is_zipfile(zipFileName):
      print('ERROR: ' + zipFileName + ' is not a valid zip file.')
      sys.exit(1)
   print('Extracting...')
   wd = os.getcwd()
   os.chdir(folder)

   archive = zipfile.ZipFile('.'+fileName, mode='r')
   archive.extractall()
   archive.close()
   os.chdir(wd)
   print('Successfully retrieved rotated MNIST dataset.')

def settings(args):
   #print('Running settings fn')
   # Download MNIST if it doesn't exist
   args.dataset = 'rotated_mnist'
   if not os.path.exists(args.data_dir + '/mnist_rotation_new.zip'):
      download2FileAndExtract("https://www.dropbox.com/s/0fxwai3h84dczh0/mnist_rotation_new.zip?dl=1",
         args.data_dir, "/mnist_rotation_new.zip")
   # Load dataset
   mnist_dir = args.data_dir + '/mnist_rotation_new'
   train = np.load(mnist_dir + '/rotated_train.npz')
   valid = np.load(mnist_dir + '/rotated_valid.npz')
   test = np.load(mnist_dir + '/rotated_test.npz')
   data = {}
   if args.combine_train_val:
      data['train_x'] = np.vstack((train['x'], valid['x']))
      data['train_y'] = np.hstack((train['y'], valid['y']))
   else:
      data['train_x'] = train['x']
      data['train_y'] = train['y']
      data['valid_x'] = valid['x']
      data['valid_y'] = valid['y']
   data['test_x'] = test['x']
   data['test_y'] = test['y']

   
   # Other options
   if args.default_settings:
      args.n_epochs = 2000
      args.batch_size = 46
      args.learning_rate = 0.076
      args.std_mult = 0.7
      args.delay = 12
      args.phase_preconditioner = 7.8
      args.filter_gain = 2
      args.filter_size = 5
      args.n_rings = 4
      args.n_filters = 8
      args.display_step = len(data['train_x'])/64
      args.is_classification = True
      args.dim = 28
      args.crop_shape = 0
      args.n_channels = 1
      args.n_classes = 10
      args.lr_div = 10.
      args.model_path = './models/'
      args.train_mode = True
      args.load_pretrained = False
      args.pretrained_model = './models/model_3235.pth'

   args.log_path = add_folder('./logs')
   args.checkpoint_path = add_folder('./checkpoints') + '/model.ckpt'
   return args, data

def add_folder(folder_name):
   if not os.path.exists(folder_name):
      os.mkdir(folder_name)
      print('Created {:s}'.format(folder_name))
   return folder_name


def minibatcher(inputs, targets, batchsize, shuffle=False):
   assert len(inputs) == len(targets)
   if shuffle:
      indices = np.arange(len(inputs))
      np.random.shuffle(indices)
   for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
      if shuffle:
         excerpt = indices[start_idx:start_idx + batchsize]
      else:
         excerpt = slice(start_idx, start_idx + batchsize)
      yield inputs[excerpt], targets[excerpt]

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

def main(args):

   ##### SETUP AND LOAD DATA #####
   args, data = settings(args)

   # choosing the device to run the model
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(device)
   # creating train_loader and valid_loader
   train_dataset = RotMNISTDataset(data['train_x'], data['train_y'])
   valid_dataset = RotMNISTDataset(data['valid_x'], data['valid_y'])
   print(len(train_dataset), len(valid_dataset))

   trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
   validloader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True)


   # gathering parameters for training
   lr = args.learning_rate

   model = DeepMNIST(args).to(device)

   if args.load_pretrained:
      model.load_state_dict(torch.load(args.pretrained_model))

   # print model parameters count
   pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   print('Total trainable params : ', pytorch_total_params)

   # Optimizer
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

   #lr scheduler
   #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
   lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=lr, step_size_up=20, mode='triangular2', cycle_momentum=False)
   
   
   lossfn = torch.nn.CrossEntropyLoss() # defining the loss function
   print('No. of batches : ', len(trainloader))
   print('Starting the training......')

   val_best = 0.0

   if not args.train_mode:
      args.n_epochs = 1
   # Starting to train the model
   for epoch in range(args.n_epochs):

      if args.train_mode:
         # Training phase
         model.train()
         if epoch % 100 == 0:
            lr_scheduler.step()

         epoch_loss = 0
         epoch_acc = 0

         correct = 0
         for idx, batch in enumerate(trainloader):
            images = batch[0]
            labels = batch[1]

            # Transfer to GPU
            images, labels = images.to(device), labels.to(device)
            labels = labels.type(torch.cuda.LongTensor if torch.cuda.is_available() \
                                                         else torch.LongTensor)

            optimizer.zero_grad()
            logits = model(images)
            correct += (torch.argmax(logits, dim=1).type(labels.dtype)==labels).sum().item()

            #loss = lossfn(logits, labels)
            loss = F.nll_loss(F.log_softmax(logits, dim=1), labels, reduction='sum')
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

         epoch_acc = correct / (len(trainloader)*args.batch_size)
         epoch_loss /= len(train_dataset)
         current_lr = lr_scheduler.get_lr()[0]
         print('Epoch: ', epoch+1, '; lr: ', current_lr, '; Loss: ', epoch_loss, '; Train Acc: ', epoch_acc, end = " ")

      # Validation phase
      model.eval()
      with torch.no_grad():
         val_acc = 0.0
         correct = 0
         for idx, batch in enumerate(validloader):
            images = batch[0]
            labels = batch[1]

            # Transfer to GPU
            images, labels = images.to(device), labels.to(device)
            labels = labels.type(torch.cuda.LongTensor if torch.cuda.is_available() \
                                                         else torch.LongTensor)

            logits = model(images)
            correct += (torch.argmax(logits, dim=1).type(labels.dtype)==labels).sum().item()
         val_acc = correct / (len(validloader)*args.batch_size)
         if val_acc > val_best:
            val_best = val_acc
            # save the cuurrent model
            save_path = args.model_path + '/model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), save_path)

         print('; Val. Acc: ', val_acc, '; Best: ', val_best)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--data_dir", help="data directory", default='./data')
   parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
   parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool, default=False)
   main(parser.parse_args())


     
