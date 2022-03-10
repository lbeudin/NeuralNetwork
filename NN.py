# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:37:13 2021

@author: Léa
"""


import numpy as np
import pandas as pd
import random 
import math

import configparser     
from pylab import mpl, plt
import quandl as q
import statsmodels.api as sm
from scipy.ndimage.interpolation import shift
  
class neuralNetwork:
    def __init__(self,layerSizeList,learningRate, activationFunction, derivativeFunction):
        self.layerSizeList = layerSizeList
        self.biases = [np.random.randn(y, 1) for y in layerSizeList[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layerSizeList[:-1], layerSizeList[1:])]
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        self.derivativeFunction = derivativeFunction
        self.num_layers = len(layerSizeList)
        
    def forwardProp(self, x_input):
      
        # forward propagation:
        fwdprop = [np.array(x_input).reshape(-1,1)] 
        activ = []
        for biase, weight in zip(self.biases, self.weights):
            activ.append(np.dot(weight, fwdprop[-1]) + biase)
            fwdprop.append((self.activationFunction(activ[-1])))
        return fwdprop[-1]

    def backprop(self, x_input, y_output):
        new_biases = [np.zeros(biase.shape) for biase in self.biases]
        new_weights = [np.zeros(weight.shape) for weight in self.weights]
    
        # forward propagation:
        fwdprop = [np.array(x_input).reshape(-1,1)] 
        activ = []
        for biase, weight in zip(self.biases, self.weights):
            activ.append(np.dot(weight,fwdprop[-1]) + biase)
            fwdprop.append((self.activationFunction(activ[-1])))
    
        # backward propagation
        delta = (fwdprop[-1] - y_output) *  self.derivativeFunction(activ[-1]) 

        for l in range(1, len(self.layerSizeList)-1): 
            new_biases[-l] = delta
            new_weights[-l] = np.dot(delta, fwdprop[-l-1].transpose())
            delta = np.dot(self.weights[-l].transpose(), delta) *  self.derivativeFunction(activ[-l-1]) 
            
        new_biases[-l-1] = delta
        new_weights[-l-1] = np.dot(delta, fwdprop[-l-2].transpose())

        return (new_biases, new_weights)


    
    def updateWeights(self, subset):
        sum_new_biase = [np.zeros(b.shape) for b in self.biases]
        sum_new_weight = [np.zeros(w.shape) for w in self.weights]
        for x in subset:
            new_biase, new_weight = self.backprop(x[1:], x[0])
            sum_new_biase = [a+b for a, b in zip(new_biase, sum_new_biase)]
            sum_new_weight = [a+b for a, b in zip(new_weight, sum_new_weight)]
        self.weights = [a-(self.learningRate/len(subset))*b
                        for a, b in zip(self.weights, sum_new_weight)]
        self.biases = [a-(self.learningRate/len(subset))*b
                       for a, b in zip(self.biases, sum_new_biase)]
            
        
    
    def train(self, train_set, epochs, trainingSetSize):
        for j in range(epochs): 
            #aleatory order
            random.shuffle(train_set)
            #subset the training set into a list of small list of tuple(input,output)
            subsets = [train_set[k:k+trainingSetSize,:]
                for k in range(0, len(train_set), trainingSetSize)]
            for subset in subsets:
                self.updateWeights(subset) 

    
    def test(self, test_set):
        inputvect = np.array(test_set)
        inputx = inputvect[:,1:]
        inputy =  inputvect[:,0]
        tt = [self.forwardProp(inputx[i,:]) for i in range(0,len(inputy)) ]
        mse = [(self.forwardProp(inputx[i,:])-inputy[i])**2 for i in range(0,len(inputy))]
        res = sum(mse)/len(inputy)
        print("MSE : "+ str(res[0][0]))
        
        

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_dev(x):
    return sigmoid(x)*(1-sigmoid(x))

def multipleRegressionLinear(train_set,test_set):
    inputvect_train = np.array(train_set)
    X = inputvect_train[:,1:]
    Y =  inputvect_train[:,0]
    
    inputvect = np.array(test_set)
    inputx = inputvect[:,1:]
    inputy =  inputvect[:,0]
    reg = sm.OLS(Y,X)
    resReg = reg.fit()
    prediction = reg.predict(resReg.params,inputx )
    mse = [(prediction[i]-inputy[i])**2 for i in range(0,len(inputy))]
    result = sum(mse)/len(inputy)
    print("MSE  ",result)



plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
c = configparser.ConfigParser()
c.read('../pyalgo.cfg')
q.ApiConfig.api_key = 'input your quandl key'

#check Bitcoin history

d = q.get('BCHAIN/MKPRU')
#d['SMA'] = d.shift(periods=1,axis=0) #d['Value'].rolling(100).mean()
d['SMA'] = d.shift(periods=1,axis=0).rolling(100).mean()
d.loc['2013-1-1':].plot(title='BTC/USD exchangerate',figsize=(10, 6));
d.head()

# Construct database
#Input eurostoxx gold
d=pd.DataFrame(d)
gold = q.get('LBMA/GOLD')
d['gold'] = gold.iloc[:,0]
yield_us = q.get('USTREASURY/REALYIELD')
d['us_yield'] = yield_us.iloc[:,2]


prev =d.shift(periods=1,axis=0)
var = d/prev-1
var=var.loc['2017-1-1':]

var=var.dropna()

for i in range(0,(var.shape[0])):
    if abs(var.iloc[i,3]) == np.inf :
        var.iloc[i,3] = 0

learning =int(len(var)*0.8)
train = np.array(var.iloc[:learning,:])

test = var.iloc[learning:,:]

nn = neuralNetwork([3,4,3,4,1],2, sigmoid, sigmoid_dev)
nn.train(train,5,50)
nn.test(test)

multipleRegressionLinear(train,test)

