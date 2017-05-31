# -*- coding: utf-8 -*-
"""
Created on Mon May  8 22:38:29 2017

@author: Andrei
"""

from online_dnn_engine import OnlineDNN
from online_dnn_engine import odn_utils
from online_dnn_engine import OnlineDNNLayer

from scipy.ndimage import imread

from matplotlib import pyplot as plt

import numpy as np

if __name__ == '__main__':

    print("OnlineDNN Autoencoder test")
    
    # load images
    s_imgfile = "img_xsmall.jpg"
    np_img = imread(s_imgfile, flatten=True)
    np_X = np_img.reshape(1,np_img.size)
    img_size = np.sqrt(np_img.size) # assume square image
    np_X /= 255
    np_y = np_X
    nr_in_units = np_X.size
    nr_out_units= nr_in_units
    plt.matshow(np_X.reshape(img_size,img_size), cmap="gray")
    plt.show()
        
    # setup Autoencoder DNN
    
    test_size = 32
    h1_activ = 'direct'
    h1_units = test_size * test_size
    
    print("Input image size: {:,} B".format(nr_in_units))
    print("H1 weights: {:,} units".format(nr_in_units * h1_units))
    print("Output weights: {:,} units\n".format(h1_units * nr_out_units))
    
    
    dnn = OnlineDNN(alpha = 0.00001, Verbose=100, best_theta = True)

    InputLayer   = OnlineDNNLayer(nr_units = nr_in_units, 
                                  layer_name = 'Input Layer')
     
    EncoderLayer = OnlineDNNLayer(nr_units = h1_units, 
                                  layer_name = 'Hidden Layer #1',
                                  activation = h1_activ)
    
    OutputLayer  = OnlineDNNLayer(nr_units = nr_out_units, 
                                  layer_name = 'Output Layer',
                                  activation = 'direct')      
    
    
    dnn.AddLayer(InputLayer)
    dnn.AddLayer(EncoderLayer)
    dnn.AddLayer(OutputLayer)
    
    dnn.PrepareModel(cost_function='MSE')

    # train 
    if dnn.ModelPrepared:
        trainer = odn_utils()
        trainer.train_online_model_no_tqdm(dnn, np_X, np_y, epochs = 20)
        
        plt.plot(dnn.cost_list)
        plt.show()
    
        # display middle layer
        print("Encoder:", flush = True)
        enc = (EncoderLayer.z_array.reshape(test_size,test_size)*1000).astype(int)
        plt.matshow(enc, cmap="gray")
        plt.show()
        
        print("Standard output:", flush = True)
        yhat1 = dnn.Predict(np_X)
        yhat1 = (yhat1*1000).astype(int)
        plt.matshow(yhat1.reshape(img_size,img_size), cmap="gray")
        plt.show()
        
        if dnn.best_theta:
            print("BestTheta output:", flush = True)
            dnn.RestoreBestThetas()
            yhat2 = dnn.Predict(np_X)
            yhat2 = (yhat2*1000).astype(int)
            plt.matshow(yhat2.reshape(img_size,img_size), cmap="gray")
            plt.show()
            print("", flush = True)
            dnn.MSE(np_X,np_X)
            print("", flush = True)
    