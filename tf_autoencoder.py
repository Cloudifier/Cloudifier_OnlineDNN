# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:11:02 2017

@author: Andrei
"""


from matplotlib import pyplot as plt

import numpy as np

from  time import time as tm

import tensorflow as tf

from tqdm import tqdm

if __name__ == '__main__':

    print("TF Autoencoder test")
    
    # load images
    s_imgfile = "img_xsmall.jpg"

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    np_img = plt.imread(s_imgfile, format='jpeg')
    np_img = rgb2gray(np_img)
    np_X = np.array(np_img.reshape(1,np_img.size), dtype = np.float32)
    img_size = int(np.sqrt(np_img.size)) # assume square image
    np_X /= 255
    np_y = np_X
    nr_in_units = np_X.size
    nr_out_units= nr_in_units
    print("Source:", flush = True)
    plt.matshow(np_X.reshape(img_size,img_size), cmap="gray")
    plt.show()
    
    batch_size = 1
    encoder_x_size = 16
    nr_hidden = encoder_x_size * encoder_x_size
    epochs = 100
    
    print("Preparing Graph model...", flush = True)
    
    g = tf.Graph()
    with g.as_default():
        tf_X_train = tf.placeholder(dtype = tf.float32,
                                    shape=(batch_size, nr_in_units),
                                    name = "X_Train")
        tf_y_train = tf.placeholder(dtype = tf.float32,
                                    shape=(batch_size, nr_out_units),
                                    name = "y_Train")
        
        tf_weights_h1 = tf.Variable(tf.truncated_normal([nr_in_units, nr_hidden],
                                                        stddev = 0.01), 
                                    dtype = tf.float32,
                                    name = "Weights_H1")
        tf_bias_h1 = tf.Variable(tf.zeros([nr_hidden]),
                                 name = "Bias_H1")
        
        tf_weights_output = tf.Variable(tf.truncated_normal([nr_hidden,nr_out_units],
                                                            stddev=0.01),
                                        dtype = tf.float32,
                                        name = "Output_Weights")
        tf_bias_output = tf.Variable(tf.zeros([nr_out_units]),
                                     name = "Output_Bias")
        
        tf_a1 = tf.matmul(tf_X_train,tf_weights_h1) + tf_bias_h1
        
        tf_output = tf.matmul(tf_a1, tf_weights_output) + tf_bias_output
         
        #tf_loss = tf.nn.l2_loss(tf_output - tf_y_train)
        tf_loss = tf.reduce_mean(tf.square(tf.subtract(tf_output, tf_y_train)))
        
        tf_optimizer  = tf.train.AdamOptimizer()
        tf_opt_op = tf_optimizer.minimize(tf_loss)
        print("Graph model constructed !", flush = True)
        
    
        
    with tf.Session(graph = g) as tf_session:
        print("Initializing variables...", flush = True)
        tf.global_variables_initializer().run()
        print("TF Initialized", flush = True)
        t0=tm()

        loss_list = list()
        for epoch in tqdm(range(epochs)):
            feed_dict = {tf_X_train: np_X, tf_y_train: np_X}
            _, loss, preds = tf_session.run([tf_opt_op, tf_loss, tf_output],
                                            feed_dict = feed_dict)
            loss_list.append(loss)
        
        t1 = tm()
        print("TF Autoencoder training finished in {:.2f}s".format(t1-t0))
        plt.plot(loss_list)
        plt.show()
        encoder_layer = tf_a1.eval(feed_dict = {tf_X_train: np_X})
        print("encoder:")
        plt.matshow(encoder_layer.reshape(encoder_x_size,encoder_x_size), cmap="gray")
        plt.show()
        print("output:")
        plt.matshow(preds.reshape(img_size,img_size),cmap="gray")
        plt.show()
    
    
                      
        
        
        
        