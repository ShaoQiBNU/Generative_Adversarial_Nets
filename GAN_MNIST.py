# -*- coding: utf-8 -*-
"""
@author: shaoqi_i
"""

########## load packages ##########
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

##################### load data ##########################
mnist = input_data.read_data_sets('MNIST_sets', one_hot=True)

########## set net hyperparameters ##########
batch_size=128
learning_rate = 0.001

########## set weights and biases parameters ##########
def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

########## set Discriminator net parameters ##########
#### real img shape:28*28, placeholder ####
d_input=784
D=tf.placeholder(tf.float32,[None,d_input], name='D')

#### weights, biases ####
D_W1 = weight_var([784, 128], 'D_W1')
D_b1 = bias_var([128], 'D_b1')

D_W2 = weight_var([128, 1], 'D_W2')
D_b2 = bias_var([1], 'D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

########## set Generator net parameters ##########
###### fake imput size: 100,placeholder ##########
g_input=100
G=tf.placeholder(tf.float32,[None,g_input], name='G')

#### weights, biases ####
G_W1 = weight_var([100, 128], 'G_W1')
G_b1 = bias_var([128], 'G_B1')

G_W2 = weight_var([128, 784], 'G_W2')
G_b2 = bias_var([784], 'G_B2')

theta_G = [G_W1, G_W2, G_b1, G_b2]

########## Discriminator model ##########
def discriminator(D):
    
    ######### input: 784 #########
    ######### 1 fc layer 128#########
    D_h1 = tf.nn.relu(tf.matmul(D, D_W1) + D_b1)
    
    ######### output layer 1 #########
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    
    ######### sigmoid #########
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

########## Generator model ##########
def generator(G):
    
    ######### input: 100 #########
    ######### 1 fc layer 128 #########
    G_h1 = tf.nn.relu(tf.matmul(G, G_W1) + G_b1)
    
    ######### output layer 784 #########
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    
    ######### sigmoid #########
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

########## Generator model input:sample data ##########
G_sample = generator(G)

########## Discriminator model input: real picture ##########
D_real, D_logit_real = discriminator(D)

########## Discriminator model input: fake picture ##########
D_fake, D_logit_fake = discriminator(G_sample)

########## Discriminator model loss1: real picture ##########
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_real, labels=tf.ones_like(D_logit_real)))

########## Discriminator model loss2: fake picture ##########
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

########## Discriminator model loss_all: real+fake ##########
D_loss = D_loss_real + D_loss_fake

########## Generator model loss: fake picture ##########
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

########## model optimizer ##########
D_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=theta_D)
G_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=theta_G)

########## generate fake picture ##########
def sample_G(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

########## plot generator result ##########
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

########## generator result save path ##########
if not os.path.exists('out/'):
    os.makedirs('out/')

########## initialize variables ##########
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    i = 0
    for it in range(1000000):
        
        ########## plot generator result ##########
        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={G: sample_G(16, g_input)})  # 16*100
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
        
        ########## train sample ##########
        batch_x, _ = mnist.train.next_batch(batch_size)
        
        ########## D loss ##########
        _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={
                              D: batch_x, G: sample_G(batch_size, g_input)})
        
        ########## G loss ##########
        _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={
                              G: sample_G(batch_size, g_input)})
        
        ########## print loss ##########
        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()


