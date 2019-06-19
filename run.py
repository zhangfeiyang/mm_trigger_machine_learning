#!/usr/bin/env python3

from datetime import datetime
import time

import tensorflow as tf
import numpy as np

#X = tf.placeholder(tf.float32, [None, 784])
#Y = tf.placeholder(tf.float32, [None, 128])

import random

Dim = 5
NN = Dim*Dim*Dim 
filename = 'Data_'+str(Dim)
file0 = open(filename,'r')
lines = file0.readlines()
N = len(lines)
N = 200000
from random import shuffle

def get_train_data():

	train_data  =[]
	train_label =[]
	for i in range(0,N,2):
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
			if j%(Dim*Dim) == Dim*Dim - 1:
				data.append(Square)
		#print(np.sum(data),int(datas[-1]))
		
		
		train_data.append(data)
		train_label.append([1,0] if int(datas[-1])==1 else [0,1])
	
	return (train_data,train_label)

def get_test_data():

	test_data  =[]
	test_label =[]
	for i in range(1,N,2):
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

(train_data,train_label) = get_train_data()

(test_data,test_label) = get_test_data()

X = tf.placeholder('float',[None,Dim,Dim,Dim,1])
Y = tf.placeholder('float',[None,2])


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

#h_pool2_flat = tf.reshape(h_pool1, [-1, 5*5*10])
#h_pool2_flat = tf.reshape(h_conv1, [-1, 15*15*10])
h_pool2_flat = tf.reshape(h_conv2, [-1, 2*2*2*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([200, 2])
b_fc2 = bias_variable([2])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_conv=tf.nn.softmax(h_fc2)

cross_entropy = -tf.reduce_sum(Y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.002).minimize(cross_entropy)
#
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#sess.run([train_step,cross_entropy],feed_dict = {X:train_data, Y:train_label})
sess = tf.Session()

sess.run(tf.global_variables_initializer())

NNNN = len(train_data)
saver = tf.train.Saver()
saver.restore(sess,"./model/my_checkpoint")

for i in range(1000):
	print(i)
	for j in range(1):
		#print('hello')
#		print(i,j)
		#print(batchs_train[0][j],batchs_train[1][j])
		#for tmp in batchs_train[0][j]:
		#	print(tmp)
		#W,b = sess.run([W_conv1,b_conv1])
		start = random.choice(range(0,NNNN,int(NNNN/10)))	
		#_, Loss= sess.run([train_step,cross_entropy],feed_dict={X:train_data[start:start+int(NNNN/10)], Y:train_label[start:start+int(NNNN/10)], keep_prob:0.9})
		_, Loss= sess.run([train_step,cross_entropy],feed_dict={X:train_data[start:start+int(NNNN/10)], Y:train_label[start:start+int(NNNN/10)], keep_prob:0.9})
		#_, Loss= sess.run([train_step,cross_entropy],feed_dict={X:train_data, Y:train_label, keep_prob:0.9})
	if i%10 == 0:
		#start = random.choice(range(0,NNNN,int(NNNN/10)))	
		#Acc = sess.run(accuracy,feed_dict={X:train_data[start:start+int(NNNN/10)], Y:train_label[start:start+int(NNNN/10)],keep_prob:1.0})
		Acc = sess.run(accuracy,feed_dict={X:train_data, Y:train_label,keep_prob:1.0})
		saver.save(sess,"./model/my_checkpoint")
		try:
			print("loss %d,\t Acc%g"%(Loss,Acc))
		except:
			print(Loss)
			print(Acc)
			sess.run(tf.global_variables_initializer())
	if i%100 ==0:
		print("test accuracy %g" %sess.run( accuracy, feed_dict={X: test_data, Y: test_label, keep_prob:1.0}))
		#print("Acc %g"%(Acc))


print("test accuracy %g" %sess.run( accuracy, feed_dict={X: test_data, Y: test_label, keep_prob:1.0}))

"""
result = (sess.run( y_conv, feed_dict={X: test_data, Y: test_label}))

NNNNN = len(result)

for i in range(NNNN):
	print(result[i],test_label[i])

"""


#logits = cifar10.inference(train_data)
#logits = cifar10.inference(X)

#loss = cifar10.loss(logits, train_label)


"""
for step in range(num_iter):

	image_batch, label_batch = sess.run([test_data,test_label])	
	#image_batch, label_batch = sess.run([train_data,train_label])	
	predictions = sess.run([top_k_op])
	print(predictions)
	
	print("length of pre is "+str(len(predictions[0])))
	
	true_count += np.sum(predictions)
	print('sum prediction is '+str(np.sum(predictions)))
	
precision = true_count / total_sample_count
print(str(100*precision)+'%')
"""		
	
