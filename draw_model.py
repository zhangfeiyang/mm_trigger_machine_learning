#!/usr/bin/env python3

from datetime import datetime
import time

import tensorflow as tf
import numpy as np

import random

Dim = 5
NN = Dim*Dim*Dim 
filename = 'Data_'+str(Dim)
file0 = open(filename,'r')
lines = file0.readlines()
N = len(lines)
N = 200000
from random import shuffle



def weight_variable(shape):
	
	initial = tf.truncated_normal(shape,mean = 0,stddev = 0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	
	initial = tf.constant(0.01,shape=shape)
	return tf.Variable(initial)

def conv3d(x,W):
	
	return tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='SAME')
	
def max_pool_3x3(x):

	return tf.nn.max_pool3d(x,ksize = [1,3,3,3,1],strides=[1,2,2,2,1],padding='VALID')

X = tf.placeholder('float',[None,Dim,Dim,Dim,1])
Y = tf.placeholder('float',[None,2])

W_conv1 = weight_variable([2,2,2,1,20])
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv3d(X, W_conv1) + b_conv1)
h_pool1 = max_pool_3x3(h_conv1)

W_conv2 = weight_variable([2,2 ,2, 20,32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([2*2*2*32, 200])
b_fc1 = bias_variable([200])

h_pool2_flat = tf.reshape(h_conv2, [-1, 2*2*2*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([200, 2])
b_fc2 = bias_variable([2])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_conv=tf.nn.softmax(h_fc2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

write = tf.summary.FileWriter('.')
write.add_graph(tf.get_default_graph())

saver.restore(sess,"./model/my_checkpoint")

