# -*- coding: utf-8 -*-
"""

Fully Connected Online Deep Neural Network 

@created: 2016-10-01
@author: Andrei Ionut DAMIAN

see also module __ver__,__credits__,.. etc loaded within OnlineDNN

"""

from __future__ import print_function


import numpy as np
import pandas as pd

from scipy.special import expit
from datetime import datetime as dt
from time import time as tm




"""

OnlineDNN classes and utilities
 
"""

__author__     = "Andrei Ionut DAMIAN"
__copyright__  = "Copyright 2017, Cloudifier SRL"
__credits__    = ["Octavian Bulie","Alexandru Purdila"]
__license__    = "GPL"
__version__    = "0.1.1"
__maintainer__ = "Andrei Ionut DAMIAN"
__email__      = "damian@cloudifier.net"
__status__     = "Production"
__library__    = "Online Deep Neural Network"
__created__    = "2016-10-01"
__modified__   = "2017-03-12"
__lib__        = "ODNN"

_DEBUG = True # module level debug flag


class OnlineDNNBase:
  def __init__(self):
      # nothing special
      return
      
  def _logger(self, logstr, show = True):
      if not hasattr(self,'USE_SHORT_DATE'):
          self.USE_SHORT_DATE = True
          
      if not hasattr(self, 'log'):        
          self.log = list()
          
      nowtime = dt.now()
      if self.USE_SHORT_DATE:
          strnowtime = nowtime.strftime("[{}][%H:%M:%S] ".format(__lib__))
      else:
          strnowtime = nowtime.strftime("[{}][%H:%M:%S] ".format(__lib__))
      logstr = strnowtime + logstr
      self.log.append(logstr)
      if show:
          print(logstr, flush = True)
      return    
  
class odn_utils(OnlineDNNBase):
  
          
  def FeatureNormalize(self,X_data, method = 'z-score'):
      if method == 'z-score':
          min_val = X_data.mean(axis=0) 
          div_val = X_data.std(axis=0)
      elif method =="minmax":
          ## min-max
          min_val = X_data.min(axis=0)  
          div_val = X_data.max(axis=0)
      else:
          raise Exception("Unknown scale/norm method: "+str(method))
  
      div_val[div_val == 0] = 1.
      
      X_norm = X_data - min_val
      X_norm = np.array(X_norm,dtype = float) / div_val
          
      return X_norm, min_val, div_val
  
  def TestDataNormalize(self, X_test, min_val,div_val):
      X_norm = X_test - min_val
      X_norm = np.array(X_norm,dtype = float) / div_val
  
      return X_norm

  def loaddata(self, file):
      return pd.read_csv(file)

  #
  # Kappa: duplicated and generalized from OnlineClassifier version
  #
  def Kappa(self,y_pred,y_truth, classes):
      nr_classes = len(classes)
      classes = list(classes)
      TP = np.zeros(shape=(nr_classes))
      FP = np.zeros(shape=(nr_classes))
      TN = np.zeros(shape=(nr_classes))
      FN = np.zeros(shape=(nr_classes))
      class_pred = np.zeros(shape=(nr_classes))
      class_real = np.zeros(shape=(nr_classes))

      for (i,c_class) in zip(range(nr_classes),classes):           
          TP[i] = np.logical_and( y_pred == c_class, y_truth == c_class ).sum()
          TN[i] = np.logical_and( y_pred != c_class, y_truth != c_class ).sum()
          FP[i] = np.logical_and( y_pred == c_class, y_truth != c_class ).sum()
          FN[i] = np.logical_and( y_pred != c_class, y_truth == c_class ).sum()
          class_pred[i] = TP[i] + FP[i]
          class_real[i] = TP[i] + FN[i]
          
      all_ex = TP[0]+TN[0]+FP[0]+FN[0]
      observed_accuracy = np.sum(TP) / all_ex
      expected_accuracy = (np.sum(class_pred*class_real) / all_ex) / all_ex
      kappa = (observed_accuracy - expected_accuracy) / \
              (1 - expected_accuracy)     
      # conf_matrix !!!
      return kappa

  #
  # ROC: duplicated and generalized from OnlineClassifier version
  #
  def ROC(self,y_prc,y_label, labels):
      nr_labels = len(labels)
      if y_label.ndim>1:
          y_label_list=y_label[:,0]
      thresholds = np.linspace(1, 0, 101)
      if nr_labels == 2:
          nr_ROCs = 1
      else:
          nr_ROCs = nr_labels
 
      TPR = np.zeros(shape=(101,nr_ROCs))
      FPR = np.zeros(shape=(101,nr_ROCs))
      AUC = np.zeros(shape=(nr_ROCs))

      for cROC in range(nr_ROCs):            
          if nr_ROCs==1:
              c_label=1
          else:
              c_label = labels[cROC]
          for i in range(101):
              c_thr = thresholds[i]
              # Classifier / label agree and disagreements for current threshold.
              if i==50:
                  k=1
              TP = np.logical_and( y_prc[:,cROC] > c_thr, y_label_list==c_label ).sum()
              TN = np.logical_and( y_prc[:,cROC] <=c_thr, y_label_list!=c_label ).sum()
              FP = np.logical_and( y_prc[:,cROC] > c_thr, y_label_list!=c_label ).sum()
              FN = np.logical_and( y_prc[:,cROC] <=c_thr, y_label_list==c_label ).sum()
          
              # Compute false positive rate for current threshold.
              FPR[i,cROC] = FP / float(FP + TN)
          
              # Compute true  positive rate for current threshold.
              TPR[i,cROC] = TP / float(TP + FN)
          
          # compute the AUC score for the ROC curve using the trapezoidal method
          AUC[cROC] = 0.
          for i in range(100):
              AUC[cROC] += (FPR[i+1,cROC]-FPR[i,cROC]) * (TPR[i+1,cROC]+TPR[i,cROC])
              
          AUC[cROC] *= 0.5
      
  
      return TPR,FPR, AUC
  
  
  ##
  ## train_online_classifier() simulates a real life 
  ## feed of data to our OnlineClassifier
  ## cross-validation is used to obtain best J(Theta)
  ## 
  def train_online_classifier(self, clf, 
                              X_train,y_train, 
                              X_cross = None,y_cross = None,
                              batch_size=1,
                              epochs = 1):    
      t0 = tm()
      nr_examples = X_train.shape[0]
      nr_batches = nr_examples // batch_size
      from tqdm import tqdm
      for epoch in tqdm(range(epochs)):
          for i in range(nr_batches): 
              xi = X_train[(i*batch_size):((i+1)*batch_size),:]
              yi = y_train[(i*batch_size):((i+1)*batch_size)]
              clf.OnlineTrain(xi,yi,X_cross=X_cross,y_cross=y_cross)
      t1 = tm()
      tdelta = t1-t0
      clf._logger("Training {} epochs finished in {:.2f}s".format(epochs,tdelta))
      clf.DumpLog()
      return clf


  ##
  ## train_online_model() simulates a real life 
  ## feed of data to our OnlineClassifier
  ## cross-validation is used to obtain best J(Theta)
  ## 
  def train_online_model_no_tqdm(self, clf, 
                         X_train,y_train, 
                         X_cross = None,y_cross = None,
                         batch_size=1,
                         epochs = 1):   
      
      nr_examples = X_train.shape[0]
      nr_batches = nr_examples // batch_size
      t0 = tm()
      for epoch in range(epochs):
          for i in range(nr_batches): 
              xi = X_train[(i*batch_size):((i+1)*batch_size),:]
              yi = y_train[(i*batch_size):((i+1)*batch_size)]
              clf.OnlineTrain(xi,yi,X_cross=X_cross,y_cross=y_cross)
      t1 = tm()
      tdelta = t1-t0
      clf._logger("Training {} epochs finished in {:.2f}s".format(epochs,tdelta))
      return clf


  ##
  ## ck_train_online_classifier() simulates a real life 
  ## feed of data to our OnlineClassifier
  ## 
  ## 
  def ck_train_online_classifier(self, clf, 
                                 X_train,y_train, 
                                 X_cross = None,y_cross = None,
                                 batch_size=1):    
      nr_examples = X_train.shape[0]
      nr_batches = nr_examples / batch_size
      for i in range(nr_batches): 
          xi = X_train[(i*batch_size):((i+1)*batch_size),:]
          yi = y_train[(i*batch_size):((i+1)*batch_size)]
          clf.OnlineTrain(xi,yi,X_cross=X_cross,y_cross=y_cross)
      return clf


valid_activations = ['','direct','sigmoid','relu','tanh','softmax']

valid_cost = ['cross_entropy','MSE']

valid_layers = ['','input','hidden','output']

class OnlineDNNLayer(OnlineDNNBase):
  
  def __init__(self, nr_units, layer_name = '', 
               PreviousLayer = None, NextLayer = None,
               activation = '', layer_type = '', output = False
               ):
      
      self.layer_name = layer_name        
      self.layer_id = -1
      self.layer_type = layer_type        
      self.nr_units = nr_units
      if not (layer_type in valid_layers):
          raise Exception("[OnlineDNNLayer:" + str(self.layer_id)+" ERROR]"+
                          " unknown layer type: " + layer_type)
      if not (activation in valid_activations):
          raise Exception("[OnlineDNNLayer:" + str(self.layer_id)+" ERROR]"+
                          " unknown activation: " + activation)
      self.activation = activation
      self.PreviousLayer = PreviousLayer
      self.NextLayer = NextLayer
      self.Theta = None 
      self.a_array = None
      self.z_array = None
      self.delta_calc = False
      self.delta = 0
      self.is_output = output
      self.y_ohm = None
      self.y_lbl = None
      self.cost_function = ''
      
      self.ThetaSaved = None
      self.BestTheta = None
      
      self.gradient = None
      self.momentum = None
      self.numericg = None
      
      self.step = -1

      return
      
  
  def IsComputingOk(self,Z):
      res = True        
      nr = np.sum(np.isinf(Z))
      nr +=np.sum(np.isnan(Z))
      if nr>0 :
          res = False
      return res
      
  def SaveTheta(self):
      
      self.ThetaSaved = np.array(self.Theta)
      return

  def RestoreTheta(self):
      if self.ThetaSaved  is not None:
          self.Theta = self.ThetaSaved
      return

  def SaveBestTheta(self):
      
      self.BestTheta = np.array(self.Theta)
      return
  
  def RestoreBestTheta(self):
      if self.BestTheta  is not None:
          self.Theta = self.BestTheta
      return

  
      
  def LongDescribe(self):
      
      res1 = self.Describe() 
  
      res = '\n y_OHM:'
      res += '\n {}'.format(self.y_ohm)
      res += '\n A_array:'
      res += '\n {}'.format(self.a_array)
      res += '\n Theta:'
      res += '\n {}'.format(self.Theta)
      res += '\n delta:'
      res += '\n {}'.format(self.delta)
      res += '\n Gradient:'
      res += '\n {}'.format(self.gradient)
      
      self._logger(res)
      
      return (res1+res)

  def Describe(self):
      
      res  = ' Layer:[{}]'.format(self.layer_id)
      res += ' Name:[{}]'.format(self.layer_name)
      res += ' Type:[{}]'.format(self.layer_type)
      res += ' Act:[{}]'.format(self.activation)
      res += ' Units:[{}]'.format(self.nr_units)
      
      self._logger(res, show = False)
      
      return res
      

  def GetOutputLabels(self):
      act = self.activation
      if   act == 'softmax':
          y_pred = np.argmax(self.a_array,axis=1)
      elif act == 'direct':
          y_pred = self.a_array
      else:
          raise Exception("[OnlineDNNLayer:" + str(self.layer_id)+" ERROR]"+
                          " unknown output computation: " + act)
      return y_pred
  
  def SetLabels(self, y):
      act = self.activation
      if   act == 'softmax':
          # generate one-hot-matrix
          labels = self.nr_units
          nr_obs = y.shape[0]
          ohm = np.zeros((nr_obs,labels))
          ohm[np.arange(nr_obs),y.astype(int)] = 1
          self.y_ohm = ohm
          self.y_lbl = y
      elif act == 'direct':
          self.y_lbl = y
      else:
          raise Exception("[OnlineDNNLayer:" + str(self.layer_id)+" ERROR]"+
                          " unknows outpu computation: " + act)
      return
      
 
  def ThetaNoBias(self):
      
      return  self.Theta[1:,:]       
      
  
  def Activate(self):
      act = self.activation
      if np.count_nonzero(self.z_array)==0:
          raise Exception("[OnlineDNNLayer:"+str(self.layer_id)+" ERROR]"+
                          " zero input received for layer:")
      if   act == 'sigmoid':
          self.a_array = self.sigmoid(self.z_array)
      elif act == 'relu':
          self.a_array = self.relu(self.z_array)
      elif act == 'tanh':
          self.a_array = self.tanh(self.z_array)
      elif act == 'softmax':
          self.a_array = self.softmax(self.z_array)
      elif act == 'direct':
          self.a_array = self.z_array
      else:
          raise Exception("[OnlineDNNLayer:" + str(self.layer_id)+" ERROR]"+
                          " unknown activation !")
      
      return
  
  
  def J(self):
      if self.y_ohm  is not None:
          Jt = self.CostFunction(self.y_ohm)
      else:
          Jt = self.CostFunction(self.y_lbl)
          
      return Jt
      
  def CostFunction(self,y_labels):
      
      if   self.cost_function == 'cross_entropy':
          J = self.log_loss(y_labels,self.a_array)
      elif self.cost_function == 'MSE':        
          J = self.MSE(y_labels,self.a_array)
      else:
          raise Exception("[OnlineDNNLayer:" + str(self.layer_id)+" ERROR]"+
                          " unknown Cost !")
      return J
      
      
  def DCostFunction(self):
      if self.y_ohm  is not None:
          y_labels = self.y_ohm
      else:
          y_labels = self.y_lbl
          
      if   self.cost_function == 'cross_entropy':
          deriv = self.Dlog_loss(y_labels,self.a_array)
      elif self.cost_function == 'MSE':        
          deriv = self.DMSE(y_labels,self.a_array)
      else:
          raise Exception("[OnlineDNNLayer:" + str(self.layer_id)+" ERROR]"+
                          " unknown Cost !")

      return deriv
 

  def GetDerivative(self):
      act = self.activation

      if   act == 'sigmoid':
          deriv = self.Dsigmoid(self.z_array)
      elif act == 'relu':
          deriv = self.Drelu(self.z_array)
      elif act == 'tanh':
          deriv = self.Dtanh(self.z_array)
      elif act == 'direct':
          deriv = 1
      else:
          raise Exception("[OnlineDNNLayer:" + str(self.layer_id)+" ERROR]"+
                          " unknown activation !")
      
      return deriv
      
  
  def InitLayer(self, PreviousLayer):


      self.nr_weights = 0
      
      if PreviousLayer == None:
          # return if first layer
          return
      self.PreviousLayer = PreviousLayer
      nr_prev = self.PreviousLayer.nr_units
      nr_curr = self.nr_units
      
      self.nr_weights = (nr_prev+1) * nr_curr
      
      ## initialize Theta
      ## size is InLayer+1 X OutLayer (+1 for bias)

      self.Theta = np.array(np.random.uniform(low=-0.005, 
                                     high=0.005, 
                                     size=(nr_prev+1,nr_curr),
                                     ), dtype = np.float32)
      
      return
      
  def sigmoid(self,z):
      return expit(z)
      
  def Dsigmoid(self,z):
      return self.sigmoid(z)*(1-self.sigmoid(z))
  
  def softmax(self,z): 
      # z is MxK where M=observation K=classes
      # first shift the values of f so that the 
      # highest number is 0:
      z -= np.max(z) 
      
      ez = np.exp(z)
      
         
      p = (ez.T / np.sum(ez, axis=1)).T 
          
      if not self.IsComputingOk(p):
          print("z={}".format(z))
          print("ez={}".format(ez))
          print("p={}".format(p))
          print("z nan ={}".format(np.isnan(z).sum()))
          print("ez nan ={}".format(np.isnan(ez).sum()))
          print("p nan ={}".format(np.isnan(p).sum()))
          print("z inf ={}".format(np.isinf(z).sum()))
          print("ez inf ={}".format(np.isinf(ez).sum()))
          print("p inf ={}".format(np.isinf(p).sum()))
          raise Exception('INF/NAN value in softmax step {}'.format(
                  self.step))            
      return p
  
  def Dsoftmax(self,y,y_pred):
      return self.dlog_loss(y,y_pred)
      
  def log_loss(self,y,y_pred):
      ##
      ## Generalized cross-entropy. y input is a OneHot matrix
      ##
      J_matrix = y*np.log(y_pred)
      m = y_pred.shape[0]
      if not self.IsComputingOk(J_matrix):
          raise Exception('INF/NAN value in log_loss step {}'.format(
                  self.step))
      J =-np.sum(J_matrix)
      J /= m
      return J
  
  def Dlog_loss(self,y,y_pred):
      t = y_pred - y
      return t
  
  def MSE(self,y,y_pred):
      m = y_pred.shape[0]
      
      J = np.sum((y-y_pred)**2)
      J = J / (2 * m)
      return J
      
  def DMSE(self,y,y_pred):
      m = y_pred.shape[0]
      
      J = (y_pred-y)
      J = J / (m)
      return J
  
  def relu(self,z):
      a = np.array(z)
      np.maximum(a,0,a)
      return a
  
  def Drelu(self,z):
      a = (z > 0).astype(int)
      return a
      
  def tanh(self,z):
      a = np.tanh(z)
      return a

  def Dtanh(self,z):
      a = 1 - np.tanh(z)**2
      return a
      
  def FProp(self, inp_array):
      nr_rows = inp_array.shape[0]
      if self.PreviousLayer == None:
          # layer is input
          # just add bias
          self.a_array = inp_array
      else:
          self.z_array = self.PreviousLayer.a_array.dot(self.Theta)
          self.Activate()
          
      if self.layer_type != 'output':
          # add bias if not output layer
          self.a_array = np.c_[np.ones((nr_rows,1)),self.a_array]
      return
      
  def BProp(self):
      ## compute derivative of activation or cost
      prev_act = self.PreviousLayer.a_array
      m =  prev_act.shape[0]
      if self.NextLayer == None:
          # this must be output layer then !
          self.delta = self.DCostFunction()
      else:
          ## now lets handle hidden layers
          deriv = self.GetDerivative()
          next_layer_delta = self.NextLayer.delta
          next_layer_ThetaNoBias = self.NextLayer.ThetaNoBias()
          ## first compute current layer delta            
          self.delta = next_layer_delta.dot(next_layer_ThetaNoBias.T)*deriv

      ## now compute current layer gradient
      self.gradient = prev_act.T.dot(self.delta) / m
      ## now add regularization
      ##
      
      return
      
      

class OnlineDNN(OnlineDNNBase):
  def __init__(self, 
               output_labels = None,  # list of output labels
               alpha = 0.1,           # learning rate
               momentum = 0.9,
               Verbose = 1,           # default minimal debug info
               Name = 'TestNet',      # default net name
               best_theta = False,    # will keep best loss weights if true
               ):
      
      self.best_theta = best_theta
      
      self.ModelPrepared = False # first un-prepare the model

      self.__author__     = __author__
      self.__version__    = __version__
      self.__library__    = __library__
      self.__lib__        = __lib__
      self._logger("{} ver: {}".format(self.__library__,
            self.__version__))
      self.Layers = list()
      self.Labels = output_labels
      self.alpha = alpha

      self.Verbose = Verbose
      self.Step = 1
      self.cost_list = list()
      self.Name = Name
      self.nr_layers = 0
      self.momentum_speed = momentum

      return
      
  def DumpLog(self):
      print("\n".join(self.log), flush = True)
      return
  
  
  def Describe(self):
      res =''
      res+='OnlineDNN v{}'.format(self.__version__)
      res+=" Name: '{}'".format(self.Name)
      res+=" Layer:{}".format(self.nr_layers)
      self._logger(res, show = False)
      res2 = "Layers:"
      for i in range(self.nr_layers):
          res2 += "\n  "+self.Layers[i].Describe();
                             
      self._logger(res2, show = True)
      
      return res
  
  def DebugInfo(self, Value, lvl=0):
      if lvl>self.Verbose:
          return
      text = ""
      text += str(Value)
      if self.Verbose>=10:
          show = True
      else:
          show = False
      self._logger(text,show = show)
      return
      
  def ComputeNumericalGradient(self, x_batch,y_batch):
      
      nr_layers = len(self.Layers)
      e = 1e-4
      
      for lyr in range(nr_layers):
          CurLayer = self.Layers[lyr]

          # first save theta
          CurLayer.SaveTheta()
          
          CurLayer.numericg = np.zeros(CurLayer.Theta.shape)

          nr_rows,nr_cols = CurLayer.Theta.shape
          for i in range(nr_rows):
              for j in range(nr_cols):
                  initVal = CurLayer.Theta[i,j]
                  CurLayer.Theta[i,j] = initVal + e
                  loss1 = self.ComputeCost(x_batch,y_batch)
                  CurLayer.Theta[i,j] = initVal - e
                  loss2 = self.ComputeCost(x_batch,y_batch)
                  CurLayer.Theta[i,j] = initVal
                  gradAprox = (loss1 - loss2) / (2.0 * e)
                  CurLayer.numericg[i,j] = gradAprox
              
                  

          # now restore theta            
          CurLayer.RestoreTheta()
      return
  
  def BackProp(self):
      nr_layers = len(self.Layers)
      # first compute gradients
      for i in range(nr_layers-1,0,-1):
          self.Layers[i].BProp()
          
      return
      
  def ForwProp(self, x_batch):
      nr_layers = len(self.Layers)
      for i in range(nr_layers):
          self.Layers[i].FProp(x_batch)
      return
      
  def ComputeCost(self, x_batch, y_batch):
      nr_layers = len(self.Layers)
      # forward propagation
      self.ForwProp(x_batch)
      OutputLayer = self.Layers[nr_layers-1]
      OutputLayer.SetLabels(y_batch)
      
      J = OutputLayer.J()
      
      return J
      
  
  
  ##
  ## perform 1 step of stohastic gradient descent
  ##
      
  def SGDStep(self, x_batch, y_batch):
      
      if not self.ModelPrepared:
          raise Exception("[OnlineDNN ERROR] Model not prepared!") 
          return
      nr_layers = len(self.Layers)
      alpha = self.alpha
      
      # forward propagation
      self.ForwProp(x_batch)
      
      # set training labels
      OutputLayer = self.Layers[nr_layers-1]
      OutputLayer.step = self.Step
      OutputLayer.SetLabels(y_batch)
      # generate current fprop training predictions
      y_preds = OutputLayer.GetOutputLabels()
      
      J = OutputLayer.J()   
      acc = np.sum(y_batch==y_preds)/float(y_preds.shape[0])
      stp = self.Step
      
      if self.Verbose>=10:
          self.DebugInfo('[TRAIN Step:{:05d}] Acc:{:.2f} loss:{:.2f}'.format(
                          stp,acc,J),
                         lvl=1)
          
          d1_slice = y_batch.reshape(y_batch.size)[:3]
          d2_slice = y_preds.reshape(y_preds.size)[:3]
          self.DebugInfo('        yTru:{}'.format(d1_slice),lvl=2)
          self.DebugInfo('        yHat:{}'.format(d2_slice),lvl=2)
      else:            
          if (stp % 100) == 0:
              self.DebugInfo('[TRAIN Step:{:05d}] Acc:{:.2f} loss:{:.2f}'.format(
                              stp,acc,J),
                             lvl=1)
              d1_slice = y_batch.reshape(y_batch.size)[:3]
              d2_slice = y_preds.reshape(y_preds.size)[:3]
              self.DebugInfo('        yTru:{}'.format(d1_slice),lvl=2)
              self.DebugInfo('        yHat:{}'.format(d2_slice),lvl=2)
      

      if (len(self.cost_list)>1) and self.best_theta:
          if J<min(self.cost_list):
              self.DebugInfo("Found best params so far!")
              self.SaveBestThetas()

      self.cost_list.append(J)

      # and now back propagation
      self.BackProp()
              
      # now update Thetas
      for i in range(nr_layers-1,0,-1):
          grad = self.Layers[i].gradient
          momentum = self.Layers[i].momentum
          if not (momentum is None):
              momentum = self.Layers[i].momentum*self.momentum_speed
              momentum = momentum + grad
          else:
              momentum = grad
          self.Layers[i].Theta = self.Layers[i].Theta  - alpha * momentum  
          self.Layers[i].momentum = momentum
          
      self.Step += 1    
      return
      
      
      
  ##
  ## Add Layers to current model
  ##
  def AddLayer(self, NewLayer):

      nr_layers = len(self.Layers)
      if nr_layers == 0:
          # input layer
          NewLayer.layer_type = 'input'
      elif self.Layers[nr_layers-1].layer_type == 'output':
          raise Exception("[OnlineDNN ERROR] Cannot add layer after output!")

      if NewLayer.layer_type=='':
          NewLayer.layer_type = 'hidden'
          if NewLayer.activation == 'softmax':
              NewLayer.layer_type = 'output'
                  
      self.Layers.append(NewLayer)
      return
  
  def SaveBestThetas(self):
      self.DebugInfo("Saving best params...",1)
      self.BestStep = self.Step
      for i in range(1,self.nr_layers):
          self.Layers[i].SaveBestTheta()
      return
  
  def RestoreBestThetas(self):
      self.DebugInfo("Restoring best params from step {}...".format(self.BestStep),1)
      for i in range(1,self.nr_layers):
          self.Layers[i].RestoreBestTheta()
      return
  
  
  def PrepareModel(self, cost_function = 'cross_entropy' ):
      nr_layers = len(self.Layers)
      self.nr_layers = nr_layers
      if nr_layers == 0:
          raise Exception("[OnlineDNN ERROR] Zero layers !")
      elif nr_layers <3:
          raise Exception("[OnlineDNN ERROR] Nr. layers <3")
     
      self.nr_weights = 0
      # first check model capacity and generate best thetas
      for i in range(1,nr_layers):
          cunits = self.Layers[i].nr_units
          punits = self.Layers[i-1].nr_units
          self.nr_weights += (punits+1)*(cunits)

      model_size_MB = self.nr_weights*4/(1024*1024)
          
      self._logger("Model capacity: {:,} weights, {:,.2f}MB".format(
              self.nr_weights,
              model_size_MB))
      if (model_size_MB>4000):
          self._logger("Model requires to much memory, please optimize!")
          return False
      #
      
      
      PrevLayer = None
      for i in range(nr_layers):
          self.Layers[i].layer_id = i
          self.Layers[i].InitLayer(PrevLayer)
          PrevLayer = self.Layers[i]
          if i<(nr_layers-1):
              self.Layers[i].NextLayer = self.Layers[i+1]
              
      
      ## force output for last layer
      self.Layers[nr_layers-1].is_output = True
      self.Layers[nr_layers-1].layer_type = 'output'
      self.Layers[nr_layers-1].cost_function = cost_function
      if self.Verbose>0:
          self.Describe()     

      self.ModelPrepared = True            
      return True
      
  def OnlineTrain(self, xi, yi, X_cross = None, y_cross = None):
      self.SGDStep(xi,yi)
      return
  
  def Predict(self, x):

      self.ForwProp(x)
      nr_layers = len(self.Layers)        
      OutputLayer = self.Layers[nr_layers-1]
      # generate current fprop training predictions
      y_preds = OutputLayer.GetOutputLabels()        
      return y_preds
  
  def MSE(self,x,y):
      yhat = self.Predict(x)
      mse = np.sum((yhat-y)**2)
      
      if self.Verbose>=10:
          self.DebugInfo('MSE Eval: {:.2f}'.format(mse))           
          d1_slice = y.reshape(y.size)[:3]
          d2_slice = yhat.reshape(yhat.size)[:3]
          self.DebugInfo('        yTru:{}'.format(d1_slice),lvl=2)
          self.DebugInfo('        yHat:{}'.format(d2_slice),lvl=2)        
      return mse
      
"""
  END OnlineDNN Engine
  
"""



if __name__ == '__main__':

    print("OnlineDNN engine.")
