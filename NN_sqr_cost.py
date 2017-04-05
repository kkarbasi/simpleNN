# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 22:08:08 2016

@author: Kaveh
"""
import numpy as np
import random
import cPickle as pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.io
from sklearn import preprocessing as pp
import time


class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 785
        self.hiddenLayerSize = 201
        self.outputLayerSize = 10
        
        self.W1 = 0.01 * np.random.randn(self.inputLayerSize , self.hiddenLayerSize) #3 by 3
        self.W2 = 0.01 * np.random.randn(self.hiddenLayerSize  , self.outputLayerSize) #3 by 1
        
        
    def forward(self , X):
        self.z2 = np.dot(X , self.W1)
        self.a2 = self.tanh(self.z2)
        self.a2[0][-1] = 1.0
        self.z3 = np.dot(self.a2 , self.W2)
        yHat = self.sigmoid(self.z3)
        
        return yHat
    
    #backward:    
    def Backprop(self , X , y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y - self.yHat) , self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T , delta3)
        
        delta2 = np.dot(delta3 , self.W2.T)*self.tanhPrime(self.z2)
        dJdW1 = np.dot(X.T , delta2)
        
        return dJdW1 , dJdW2

    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))

    def sigmoidPrime(self , z):
        return np.exp(-z)/((1 + np.exp(-z))**2)
        #return sigmoid(z)
        
    def tanh(self , z):
        return np.tanh(z)
    
    def tanhPrime(self , z):
        return 1 - (np.tanh(z)**2)

    def train(self , X , y , labels, rate , nepochs):
        #iterCount = 0
        costs = []
        accuracies = []
        figTicks = []
        for epoch in range(nepochs):
            shuffled_indices = range(len(X));
            np.random.shuffle(shuffled_indices)
            print 'In epoch', epoch
            for i in shuffled_indices:
                Xi = X[i].reshape(-1 , X[i].shape[0])
                
                Yi = y[i]
                
                dJdW1 , dJdW2 = self.Backprop(Xi , Yi)
                self.W1 = self.W1 - rate*dJdW1
                self.W2 = self.W2 - rate*dJdW2
				# uncomment the following if you want the plots
                '''
                iterCount += 1;
                if iterCount % ((epoch+1)*1000) == 0:
                    
                    curr_loss = self.lossFunction(X , y)
                    curr_accuracy = self.getAccuracy(X , labels).tolist()
                    
                    costs = costs + [curr_loss]
                    figTicks = figTicks + [iterCount]
                    accuracies = accuracies + curr_accuracy
                    
                    print 'Iteration:' , iterCount , '| Loss:' , curr_loss , '| Training Accuracy:' , curr_accuracy
                   
            
            if epoch%2 == 0:  
                rate = rate*0.9
                '''
        return costs , accuracies , figTicks
            # Save the state of the NN at the end of the current epoch
            #self.Save(str(epoch) + '.epoch') 
    
    def test(self , X):
        yHat = self.forward(X)
        return np.argmax(yHat)
        #return self.forward(X)

    ########################################## Testing Geadient #######################


        
    def costFunction(self , X , y):
        yH = self.forward(X)
        cost = (y - yH).dot((y - yH).T)
        #print cost
        return 0.1*cost

    def lossFunction(self , X , y):
        yH = self.forward(X)
        cost = np.sum((y - yH).T.dot(y - yH))
        #print cost
        return 0.5*cost
        
    def getParams(self):
        params = np.concatenate((self.W1.ravel() , self.W2.ravel()))
        return params
    
    def setParams(self , params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start : W1_end] , (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end : W2_end] , (self.hiddenLayerSize , self.outputLayerSize))
        
    def computeGradients(self , X , y):
        dJdW1 , dJdW2 = self.Backprop(X , y)
        return  np.concatenate((dJdW1.ravel() , dJdW2.ravel()))
        
    def getAccuracy(self , X , y):
        pred = []
        for Xi in X:
            Xi = Xi.reshape(-1 , Xi.shape[0])
            pred = pred + [self.test(Xi)]
        
        pred = np.array(pred)
        pred = pred.reshape(len(pred) , 1)
        err_rate, indices = self.benchmark(pred , y)
        return 1.0 - err_rate

    def benchmark(self , pred_labels, true_labels):
        errors = pred_labels != true_labels
        err_rate = sum(errors) / float(len(true_labels))
        indices = errors.nonzero()
        return err_rate, indices

    
    #################################### Serialization####################    
    def Load(self , filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        f.close()          
    
        self.__dict__.update(tmp_dict) 

    def Save(self , filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__,f,2)
        f.close()
    ######################################################################

    
if __name__ == "__main__":

    def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 

    def benchmark(pred_labels, true_labels):
        errors = pred_labels != true_labels
        err_rate = sum(errors) / float(len(true_labels))
        indices = errors.nonzero()
        return err_rate, indices


    def retVectorDigit(d):
        '''
        return a vector of size 10
        '''
        v = np.array([0.0]*10 , dtype = 'float')
        v[d] = 1.0
        return v

    def output_csv(labels, filepath="output_MSE_1.csv"):
        X = np.array([np.array([i+1, labels[i]]) for i in range(len(labels))])
        np.savetxt(filepath, X, delimiter=',', header='Id,Category', \
                      comments="", fmt="%d")
    
    
    VAL_SIZE = 10000;
    TRN_SIZE = 50000;    
    
    digits = scipy.io.loadmat('./hw6/dataset/train.mat');

    images = digits['train_images'];
    labels = digits['train_labels'].reshape(len(digits['train_labels']) , -1) * 1.0
    
    rand_idx = range(len(labels))
    np.random.shuffle(rand_idx)
    
    X = np.transpose(images)
    X = np.transpose(X , (0,2,1))
    X = X.reshape(len(X) , -1)
    X = np.append(X , np.ones([len(X),1]),1)*1.0

        
    y = np.ndarray( shape = (labels.shape[0] , 10))        
    
    
    for i , l in enumerate(labels):
        y[i] = retVectorDigit(l[0])
    X = pp.scale(X)    
    
    X_train = X[rand_idx[0:TRN_SIZE]]
    Y_train = y[rand_idx[0:TRN_SIZE]]
    
    X_val = X[rand_idx[-VAL_SIZE:]]
    Y_val = labels[rand_idx[-VAL_SIZE:]]    
    
    
    
    NN = Neural_Network()
    
    start = time.time()
    costs , accuracies , figTicks = NN.train(X_train , Y_train , labels[rand_idx[0:TRN_SIZE]] , 0.01 , 22)
    print 'Total training time:' , time.time() - start , 'seconds'
    # uncomment the following if you want the plots:
	'''
    plt.figure()
    plt.plot(figTicks , costs)
    plt.title('Total training loss VS iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Training Loss')
    plt.show()
    
    plt.figure()
    plt.plot(figTicks , accuracies)
    plt.title('Training accuracy VS iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Training Accuracy')
    plt.show()
    '''
    
    
    print 'Validation Accuracy:' , NN.getAccuracy(X_val , Y_val)
    print 'Training Accuracy:' , NN.getAccuracy(X_train , labels[rand_idx[0:TRN_SIZE]])
    ######################## Training Accuracy ##########################

        
        


    #######################(uncomment for) TEST  ########################################
    '''
    test_images = scipy.io.loadmat('./hw6/dataset/test.mat');
    
    X_test = test_images['test_images']

    X_test = X_test.reshape(len(X_test) , -1)
    X_test = np.append(X_test , np.ones([len(X_test),1]),1)*1.0
    X_test = pp.scale(X_test)
    pred_test = []
    for Xi in X_test:
        Xi = Xi.reshape(-1 , Xi.shape[0])
        pred_test = pred_test + [NN.test(Xi)]
    
    pred_test = np.array(pred_test)
    pred_test = pred_test.reshape(len(pred_test) , 1)
    output_csv(pred_test)
    '''

    
