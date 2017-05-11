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
    s_imgfile = "img_vsmall.jpg"
    np_img = imread(s_imgfile, flatten=True)
    np_X = np_img.reshape(-1)
    np_y = np_X
    nr_in_units = np_X.size
    nr_out_units= nr_in_units
    np_X = np_X.reshape(nr_in_units,1)
    plt.matshow(np_img, cmap="gray")
        
    # setup Autoencoder DNN
    
    test_size = 64
    h1_activ = 'direct'
    h1_units = test_size * test_size
    
    dnn = OnlineDNN()

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
        trainer.train_online_model_no_tqdm(dnn, np_X, np_y, epochs = 10)
    
        # display middle layer
        
        plt.matshow(EncoderLayer.a_array.reshape(test_size,test_size), cmap="gray")
    