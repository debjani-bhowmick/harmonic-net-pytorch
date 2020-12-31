import argparse
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rotate
sys.path.append('../')  

"""
This script contains utility functions related to creating rotations in images.
"""

def create_rot_variants(args):
    """
        This function takes a set of images in npy format, and create rotation variants for each of them.
    """
    image_set = np.load(args.data_set)

    print(image_set['x'].shape)
    w, h = 28, 28

    n_images = 100

    # Iterate over first 100 images and create variants for every 5 degrees
    ang_arr = np.linspace(0, 360, 72, endpoint=False)
    list_ang = list(ang_arr)
    
    for itr in range(n_images):
        img_mat = np.zeros((72, 784))
        print(itr)
        ctr = 0
        for rot_angle in list_ang:
            img = image_set['x'][itr,:]
            img = img.reshape(w, h)
            img = img * 255.0
            #rotate the image here
            img_rot = rotate(img, rot_angle)
            io.imsave(args.write_path+'jpgs/'+str(itr)+"_"+str(int(rot_angle)) + '.jpg', img_rot.astype(np.uint8))

            img_rot = img_rot / 255.0
            img_rot = img_rot.reshape(784,)
            img_mat[ctr,:] = img_rot
            ctr+=1
        np.savez(args.write_path+str(itr)+'.npz', img_mat)
    print('Generation process finished.')



if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--data_set", help="data directory", default='./../MNIST-rot/data/mnist_rotation_new/rotated_train.npz')
   parser.add_argument("--write_path", help="save rotated files", default='./../MNIST-rot/data/mnist_rotation_new/train_100/')


   create_rot_variants(parser.parse_args())
