
#dependencies
#only importing the bare minimum to save runtime
from tensorflow import (Variable,function,matmul,constant,GradientTape,ones)#this line is too slow
from tensorflow.random import truncated_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import relu,sigmoid
from tensorflow.math import tanh
from tensorflow import function
import numpy#this is actually a dependancy of tensorflow
import struct#for export and import
from .Exceptions import *

#import layers
from .DenseLayer import DenseLayer 
from .ConvolutionLayer import ConvolutionLayer 
from .PoolingLayer import PoolingLayer 
from .FlattenLayer import FlattenLayer

#
#Convolutional Neural Network
#
#for now only works with 3 chanells
class CNN:
    #inits key values 
    def __init__(self,inputSize,debug):
        self.debug=debug
        self.inputSize=inputSize
        self.chanells=3
        self.layers=[]#holds all layers of a neural network
        self.layerKey=[]#holds string indicators of the type of each layer
        #DENSE,CONVOLUTE,FLATTEN,POOL
        self.layerOutputShape=[]#for single processing
        #the batch size should be added to each output shape for bach processing
        #used to keep track of tests
        self.totalTrainableVariables=0

    #stride is int
    def addConvolutionLayer(self,filterSize,stride):
        #check that the previous layer is not flat
        if(len(self.layers)!=0):
            if(self.layerOutputShape[-1]==1):
                #the previous layer is flat
                #a convolutional layer can't go here
                raise(invalidLayerPlacement(True,False,True))       

       #if we are here we are good to go
        #compute output size
        if(len(self.layers)==0):
            inputShape=[self.inputSize,self.inputSize,3]
        else:
            inputShape=self.layerOutputShape[-1]
        outputShape=[
            (inputShape[0]-(filterSize//2)*2)/stride,
            (inputShape[1]-(filterSize//2)*2)/stride,
            inputShape[2]
        ] 

        myConvolutionLayer=ConvolutionLayer()
        myConvolutionLayer.newLayer(filterSize,stride)
        self.layers.append(myConvolutionLayer)
        self.layerKey.append("CONVOLUTE")
        self.layerOutputShape.append(outputShape)
        self.totalTrainableVariables+=filterSize**2

    def addPoolingLayer(self,size,stride):
        #check that the previous layer is not flat
        if(len(self.layers)!=0):
            if(self.layerOutputShape[-1]==1):
                #the previous layer is flat
                #a convolutional layer can't go here
                raise(invalidLayerPlacement(True,False,True))       

        #grab input shpae
        if(len(self.layers)==0):
            inputShape=[self.inputSize,self.inputSize,3]
        else:
            inputShape=self.layerOutputShape[-1]

        #calculate output shape same way as convolutional layer
        outputShape=[
            (inputShape[0]-(size//2)*2)/stride,
            (inputShape[1]-(size//2)*2)/stride,
            inputShape[2]
        ] 
        
        #if we are here we are good to go
        self.layers.append(PoolingLayer(size,stride))
        self.layerKey.append("POOL")
        self.layerOutputShape.append(outputShape)



    def addFlattenLayer(self):
        #grab input shpae
        if(len(self.layers)==0):
            inputShape=[self.inputSize,self.inputSize,3]
        else:
            inputShape=self.layerOutputShape[-1]
        #compute output shape
        outLen=1
        for i in inputShape:
            outLen*=i

        #doesn't technically need to follow a nonFlatLayer
        myFlattenLayer=FlattenLayer()
        self.layers.append(myFlattenLayer)
        self.layerKey.append("FLATTEN")
        self.layerOutputShape.append([outLen])

    def addDenseLayer(self,layerSize,activation):
        #grab input shpae
        if(len(self.layers)==0):
            inputShape=[self.inputSize,self.inputSize,3]
        else:
            inputShape=self.layerOutputShape[-1]

        #check input is flat
        if(len(inputShape)!=1):
            #throw
            raise(invalidLayerPlacement(False,True,False))

        myDenseLayer=DenseLayer()
        myDenseLayer.newLayer(int(inputShape[0]),layerSize,activation) 

        self.layers.append(myDenseLayer)
        self.layerKey.append("DENSE")
        self.layerOutputShape.append([layerSize])
        self.totalTrainableVariables+=layerSize*int(inputShape[0])

    #returns a list of pointers to trainable variables
    def getTrainableVariables(self):
        out=[]
        for layer in self.layers:
            out+=layer.getTrainableVariables()

        return out

    #evaluates the network for a list of inputs
    #forward propagation
    #@function
    def evaluate(self,x):
        #ensure that layers are floats
        layerVals=[x]#start layerVals with the batch
        if self.debug:
            c=0
        for layer in self.layers:#for each hidden layer and the output layer
            layerVals.append(layer.execute(layerVals[-1]))#eager execution
            if(self.debug):
                print("Finished latayer\t"+self.layerKey[c]+"\t"+str(layerVals[-1].shape))
                c+=1
        #return final layer as output layer
        return layerVals[-1]

    #train a nerual netwrok to fit the data provided
    #returns squared error
    #X must be in tensorflow format
    #only trianes for one Yindex(Yi) per training example
    def train(self,X,Y,learningRate,L2val):

        #very sorry L2 is currtently out of order I will fix this later

        #apply L2 regularization to avoid overfitting
        #this is really really important
        #regularizer=l2(L2val)#just to be clear this is tf.keras.regularizers.l2
        #regularizer(self.weights)

        #compute gradients of weights and biases
        with GradientTape() as g:
            myTrainableVariables=self.getTrainableVariables()
            g.watch(myTrainableVariables)

            #calculate error
            if self.debug:
                print("EXECUTING")
            guess=self.evaluate(X)
            #calculate error using sqared error
            if self.debug:
                print("TRAINING")
            error=0
            for i in range(len(Y)):
                for j in range(len(Y[i])):
                    error+=(guess[i][j]-Y[i][j])**2
            error=error/len(Y)

        optimizer=Adam(learningRate)
        grads=g.gradient(error,myTrainableVariables)
        optimizer.apply_gradients(zip(grads,myTrainableVariables),)
        return error

    #export currently loaded network to file
    def export(self,path):
       pass 

    #import a network of the format given above
    def importNetwork(self,path):
      pass 

    #return deepcopy of self
    #TODO
    def deepcopy(self):
        return
        #TODO 
