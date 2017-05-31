# -*- coding: utf-8 -*-
"""
Created on Mon May  8 22:35:52 2017

@author: Andrei
"""

from online_dnn_engine import OnlineDNN
from online_dnn_engine import odn_utils
from online_dnn_engine import OnlineDNNLayer


from time import time as tm



import numpy as np

if __name__ == '__main__':

    
    FULL_DEBUG = False

    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.optimizers import SGD
    from keras.utils import np_utils

    
    import sklearn.datasets as ds
    from sklearn import model_selection

    import matplotlib.pyplot as plt
    
    import random
    
    mseed = random.SystemRandom().randint(1,1000000)
    
    util = odn_utils()
    
    dataset = 'MNIST'

    if dataset == 'digits':
        #load digits
        digits = ds.load_digits()
    
        nr_obs = digits.target.size   
        nr_features = 64
        img_size = 8
        X = digits.images.reshape((nr_obs,nr_features))
        #X/= np.max(X)
        y = digits.target
        labels = digits.target_names
    elif dataset == 'iris':
        iris = ds.load_iris()
        nr_obs = iris.target.size   
        nr_features = len(iris.feature_names)
        X_raw = iris.data
        y = iris.target
        labels = iris.target_names
        norm_method = 'minmax'
        X, min_val,div_val = util.FeatureNormalize(X_raw,method=norm_method)
    elif dataset =='MNIST':
        img_size = 28
        nr_features = img_size*img_size
        dataDict = ds.fetch_mldata('MNIST Original')
        labels = np.unique(dataDict.target)
        X = dataDict.data[:-10000,]
        X = X.astype(float) / 255
        y = dataDict.target[:-10000]   
        X_test = dataDict.data[-10000:,]
        X_test = X_test.astype(float) / 255
        y_test = dataDict.target[-10000:]         
    else:
        raise Exception('Unknown dataset')
    
    
    general_seed = mseed #1234
    
    X_train, X_cross, \
    y_train, y_cross = model_selection.train_test_split(
                                            X, 
                                            y,
                                            test_size=0.15,
                                            random_state=general_seed)    
    


        
        
    
  
    in_units = nr_features    
    h1_units = int(nr_features*2.5)
    h1_activ = 'relu'
    h2_units = int(h1_units*0.5)
    #h2_activ = 'tanh'
    ou_units = np.array(labels).size
    l_rate = 0.01
    mom_speed = 0.9
    
    nr_epochs = 10
    batch_size = 1024
    nr_examples = X_train.shape[0]
    nr_batches = nr_examples // batch_size

    

    


  
    ##
    ##  BEGIN MODEL ARCHITECTURE
    ##
    dnn = OnlineDNN(output_labels = labels,
                    alpha = l_rate,
                    ve
                    )
    
    InpLayer  = OnlineDNNLayer(nr_units = in_units, 
                              layer_name = 'Input Layer')
    HidLayer1 = OnlineDNNLayer(nr_units = h1_units, 
                              layer_name = 'Hidden Layer #1',
                              activation = h1_activ)
#    HidLayer2 = OnlineDNNLayer(nr_units = h2_units, 
#                              layer_name = 'Hidden Layer #2',
#                              activation = h1_activ)
    OutLayer  = OnlineDNNLayer(nr_units = ou_units, 
                              layer_name = 'Softmax Output Layer',
                              activation = 'softmax')  
    dnn.AddLayer(InpLayer)
    dnn.AddLayer(HidLayer1)
#    dnn.AddLayer(HidLayer2)
    dnn.AddLayer(OutLayer)
    dnn.PrepareModel(cost_function = 'cross_entropy')    
    ##
    ##  END  MODEL ARCHITECTURE
    ##
    
    trainer = odn_utils()
    
    t0=tm()
    trainer.train_online_classifier(dnn,X_train,y_train,
                                    batch_size=batch_size,
                                    epochs=nr_epochs)
    t1=tm()
    tm_odnn = t1-t0

    #for ep in range(nr_epochs):
    #    for i in range(nr_batches): 
    #        xi = X_train[(i*batch_size):((i+1)*batch_size),:]
    #        yi = y_train[(i*batch_size):((i+1)*batch_size)]
    #        dnn.OnlineTrain(xi,yi,X_cross=X_cross,y_cross=y_cross)
            
    y_preds = dnn.Predict(X_cross)
    acc = np.sum(y_preds == y_cross) / float(X_cross.shape[0])*100
    #print("preds: {}".format(y_preds))
    #print("cross: {}".format(y_cross))


    ###
    ###

    ### run / compare with mnielsen with same hyperparams
    ### http://neuralnetworksanddeeplearning.com/chap3.html
    ### (and/or change output to sigmoid layer)
    
    # 1st try with Keras / TF
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, ou_units)
    Y_cross = np_utils.to_categorical(y_cross, ou_units)
    
    model = Sequential()
    model.add(Dense(output_dim=h1_units, input_dim=in_units))
    model.add(Activation("sigmoid"))
    model.add(Dense(output_dim=ou_units))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=SGD(lr=l_rate, momentum=mom_speed, nesterov=False))
    
    t0 = tm()
    model.fit(X_train, Y_train, 
              nb_epoch=nr_epochs, batch_size=batch_size,
              verbose = 1)
    t1 = tm()
    tm_keras = t1-t0
    
    y_preds_keras = model.predict(X_cross).argmax(axis = 1)
    acc_keras = np.sum(y_preds_keras == y_cross) / float(X_cross.shape[0])*100
    #print("preds: {}".format(y_preds))
    #print("cross: {}".format(y_cross))
  
    ###
    ###


    print("\nResults:")
    print('KERAS (train {:.2f}min) validation accuracy: {:.1f}%'.format(tm_keras/60,acc_keras))
    print('ODNN  (train {:.2f}min) validation accuracy: {:.1f}%'.format(tm_odnn/60,acc))
    
    plt.plot(range(len(dnn.cost_list)),dnn.cost_list)
    
    if FULL_DEBUG:
        if  (dataset == 'digits') or (dataset == ' MNIST'):
            plt.matshow((X_train[0,:]*255).reshape(img_size,img_size),  cmap=plt.cm.gray)
            plt.title('Example Y[0]:{}'.format(y_train[0]))        
        elif dataset == 'iris':
            print('Example X:{} y={}'.format(X_train[0,:],y_train[0]))
        else:
            raise Exception('Unknown dataset')
