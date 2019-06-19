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

def get_test_data():

	test_data  =[]
	test_label =[]
	for i in range(N):
		datas = lines[i].split()
		data = []
		for j in range(NN):
			value = []
			value.append(int(datas[j]))
			if j%Dim == 0:
			    Line = []
			if j%(Dim*Dim) == 0:
			    Square = []
			Line.append(value)
			if j%Dim == Dim-1:
			    Square.append(Line)
			if j%(Dim*Dim) == Dim*Dim-1:
			   data.append(Square)
		
		test_data.append(data)
		
		test_label.append([1,0] if int(datas[-1])==1 else [0,1])
	
	return (test_data,test_label)

#(train_data,train_label) = get_train_data()

(test_data,test_label) = get_test_data()



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

cross_entropy = -tf.reduce_sum(Y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
#
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#sess.run([train_step,cross_entropy],feed_dict = {X:train_data, Y:train_label})
sess = tf.Session()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

#write = tf.summary.FileWriter('.')
#write.add_graph(tf.get_default_graph())


saver.restore(sess,"./model/my_checkpoint")

Y_fit = (sess.run( y_conv, feed_dict={X: test_data, Y: test_label, keep_prob:1.0}))


N = len(test_label)
Y_real = test_label

file2 = open('result','w')
for i in range(N):
	
	file2.write(str(Y_real[i][0])+'\t'+str(Y_fit[i][0])+'\t'+str(np.sum(test_data[i]))+"\n")

file2.close()


cut = 0.6

def get_prob(cut):
	total_dark = 0
	identy_dark = 0
	for i in range(N):
		#print(Y_fit[i][0],Y_real[i][0])
		if Y_real[i][0] == 0:
			total_dark += 1
			if Y_fit[i][0] < cut:
				identy_dark += 1
	return 100*identy_dark*1.0/total_dark

for i in range(20):
	cut = 0.89 + 0.001*i	
	print("Acc",cut, get_prob(cut)," %")
	
def get_prob2(cut):
	total_dark = 0
	identy_dark = 0
	for i in range(N):
		#print(Y_fit[i][0],Y_real[i][0])
		if Y_real[i][0] == 1:
			total_dark += 1
			if Y_fit[i][0] > cut:
				identy_dark += 1
	return 100*identy_dark*1.0/total_dark

print(get_prob( 0.892))
print(get_prob2(0.892))

