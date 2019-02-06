import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import cPickle as pickle
from tqdm import tqdm
import numpy as np
import argparse
import random
import ntpath
import time
import sys
import cv2
import os

sys.path.insert(0, '../ops/')

from tf_ops import *
import data_ops
from nets import *


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--CLASS', required=True,help='checkpoint directory',type=int, default=8)
   # parser.add_argument('--DATASET',        required=False,help='The DATASET to use',      type=str,default='celeba')
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--MAX_GEN',        required=False,help='Maximum images to generate',  type=int,default=5)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   # DATASET        = a.DATASET
   OUTPUT_DIR     = a.OUTPUT_DIR
   MAX_GEN        = a.MAX_GEN
   RESULT = a.CLASS
   BATCH_SIZE = 64

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   y_dim = 10
   # placeholders for data going into the network
   z = tf.placeholder(tf.float32, shape=(100, 100), name='z')
   y = tf.placeholder(tf.float32, shape=(100, y_dim), name='y')

   # generated images
   gen_images = netG(z, y, 100)
   D_score = netD(gen_images, y, 100, 'wgan')
   
   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         raise
         exit()

   # flag for male or female
   # male = 1

   c = 0
   print 'generating data...'
   while c < MAX_GEN:
      batch_z = np.random.normal(0, 1.0, size=[100, 100]).astype(np.float32)

      # first generate an image with no attributes except male or female
      # batch_y = np.random.choice([0, 1], size=(100,y_dim))
      batch_y = np.zeros((100,y_dim))
      # batch_y[0][:] = 0
      # batch_y[0][-3] = male # make male or female
      
      # first 10 belong to class 0, then next 10 to 1, so on...
      for i in range(100):
         batch_y[i][RESULT] = 1.
         # batch_y[i][1] = 1.



      gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0]

      canvas = 255*np.zeros((380,380, 3), dtype=np.uint8)
      start_x = 2
      end_x = start_x + 32
      start_y = 8
      end_y = start_y + 32
      

      print gen_imgs.shape
      ind = 0
      for img in gen_imgs:

         img = (img+1.)
         img *= 127.5
         img = np.clip(img, 0, 255).astype(np.uint8)
         img = np.reshape(img, (32, 32, -1))
         # img = cv2.resize(img,(250,250),interpolation=cv2.INTER_CUBIC)
         # cv2.imwrite('results.png',img)
         # print start_x,end_x,start_y,end_y         
         # print img.shape

         if(ind%10 == 0):
            start_x = 10
            
         else:
            start_x = start_x + 32 + 5
            
         
         if(ind%10 == 0 and ind != 0):
            start_y = start_y + 32 + 5

         else:
            start_y = start_y
         

         end_x = start_x + 32 + 5
         end_y = start_y + 32 + 5

         canvas[start_x: start_x + 32, start_y: start_y + 32, :] = img

         ind += 1

      # misc.imsave(OUTPUT_DIR+'attributes.png', canvas)
      # disp = cv2.resize(canvas,(200,200),interpolation=cv2.INTER_CUBIC)
      # cv2.imshow('Were you looking for these?',disp)
      # cv2.waitKey(0)
      # cv2.imwrite('results.png',disp)
      misc.imsave('results.png', canvas)
      exit()
