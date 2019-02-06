'''

   This file generates celeba latent z vectors and the corresponding images such that the encoder can be trained

'''
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

RESULT = 1

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--CLASS', required=False,help='required_class',type=int)
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
   z_dim = 100
   # placeholders for data going into the network
   z = tf.placeholder(tf.float32, shape=(MAX_GEN, z_dim), name='z')
   y = tf.placeholder(tf.float32, shape=(MAX_GEN, y_dim), name='y')

   # generated images
   gen_images = netG(z, y, MAX_GEN)
   D_score = netD(gen_images, y, MAX_GEN, 'wgan')
   
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
      batch_z = np.random.normal(0, 1.0, size=[MAX_GEN, 100]).astype(np.float32)

      # first generate an image with no attributes except male or female
      # batch_y = np.random.choice([0, 1], size=(100,y_dim))
      batch_y = np.zeros((MAX_GEN,y_dim))
      
      for i in range(MAX_GEN):
         batch_y[i][RESULT] = 1.

      gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0]

      canvas = 255*np.ones((1520, 320, 3), dtype=np.uint8)
      start_x = 10
      end_x = start_x + 300
      start_y = 10
      end_y = start_y + 300
      

      print gen_imgs.shape
      
      for img in gen_imgs:

         img = (img+1.)
         img *= 127.5
         img = np.clip(img, 0, 255).astype(np.uint8)
         img = np.reshape(img, (32, 32, -1))
         img = cv2.resize(img, (300,300),interpolation = cv2.INTER_CUBIC)
         # print start_x,end_x,start_y,end_y         
         # print img.shape
         canvas[start_x:start_x + 300, start_y:start_y + 300, :] = img
         start_x = start_x + 300
         
         


      # for img in gen_imgs:
      #    img = (img+1.)
      #    img *= 127.5
      #    img = np.clip(img, 0, 255).astype(np.uint8)
      #    img = np.reshape(img, (32, 32, -1))
      #    end_x = start_x+32
      #    canvas[start_y:end_y, start_x:end_x, :] = img
      #    break

      # bald, bangs, black_hair, blond_hair, eyeglasses, heavy_makeup, male, pale_skin, smiling
      # batch_y = np.random.choice([0, 1], size=(BATCH_SIZE,9))
      # batch_y[0][:] = 0
      # batch_y[0][-3] = male # make male or female
      # for i in range(10):
      #    batch_y[0][i] = 1
      #    if male == 1 and i == 6: batch_y[0][i] = 0
      #    print batch_y[0]
      #    gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0]
      #    for img in gen_imgs:
      #       img = (img+1.)
      #       img *= 127.5
      #       img = np.clip(img, 0, 255).astype(np.uint8)
      #       img = np.reshape(img, (64, 64, -1))
      #       break
      #    end_x = start_x+64
      #    canvas[start_y:end_y, start_x:end_x, :] = img
      #    start_x += 64+10
      
      #    batch_y = np.random.choice([0, 1], size=(BATCH_SIZE,9))
      #    batch_y[0][:] = 0
      #    batch_y[0][-3] = male # make male or female

      misc.imsave(OUTPUT_DIR+'demo_results.png', canvas)
      exit()
