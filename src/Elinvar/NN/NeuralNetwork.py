
#dependencies
#only importing the bare minimum to save runtime
from tensorflow import (Variable,function,matmul,constant,GradientTape,ones)
#this line is too slow
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
from .TransposeConvolutionLayer import TransposeConvolutionLayer
from .InstanceNormalizationLayer import InstanceNormalizationLayer

#
#Neural Network - supports Convolution
#
#for now only works with 3 chanells
class NeuralNetwork:
    #inits key values 
    def __init__(self,inputSize,inputChannels,debug=False):
        self.debug=debug
        self.inputSize=inputSize
        self.inputChannels=inputChannels
        self.layers=[]#holds all layers of a neural network
        self.layerKey=[]#holds string indicators of the type of each layer
        #DENSE,CONVOLUTE,FLATTEN,POOL
        self.layerOutputShape=[]#for single processing
        #the batch size should be added to each output shape for bach processing
        #used to keep track of tests
        self.totalTrainableVariables=0

    #stride is int
    def addConvolutionLayer(self,numberOfKernels,kernelSize,stride,seed=None):
        #check that the previous layer is not flat
        if(len(self.layers)!=0):
            if(len(self.layerOutputShape[-1])==1):
                #the previous layer is flat
                #a convolutional layer can't go here
                raise(invalidLayerPlacement(True,False,True))       

       #if we are here we are good to go
        #compute output size
        if(len(self.layers)==0):
            #shape per batch item
            inputShape=[self.inputSize,self.inputSize,self.inputChannels]
            layerInputChannels=self.inputChannels
        else:
            inputShape=self.layerOutputShape[-1]
            layerInputChannels=inputShape[2]
        outputShape=[
            (inputShape[0]-(kernelSize//2)*2)/stride,
            (inputShape[1]-(kernelSize//2)*2)/stride,
            numberOfKernels
        ] 

        myConvolutionLayer=ConvolutionLayer()
        myConvolutionLayer.newLayer(kernelSize,numberOfKernels,stride,layerInputChannels,seed)
        self.layers.append(myConvolutionLayer)
        self.layerKey.append("CONVOLUTE")
        self.layerOutputShape.append(outputShape)
        self.totalTrainableVariables+=kernelSize**2*layerInputChannels*numberOfKernels
        
    def addTransposeConvolutionLayer(self,numberOfKernels,kernelSize,stride,seed=None):
        #check that the previous layer is not flat
        if(len(self.layers)!=0):
            if(self.layerOutputShape[-1]==1):
                #the previous layer is flat
                #a convolutional layer can't go here
                raise(invalidLayerPlacement(True,False,True))       

       #if we are here we are good to go
        #compute output size
        if(len(self.layers)==0):
            #shape per batch item
            inputShape=[self.inputSize,self.inputSize,self.inputChannels]
            layerInputChannels=self.inputChannels
        else:
            inputShape=self.layerOutputShape[-1]
            layerInputChannels=inputShape[2]
        outputShape=[
            (inputShape[0]+(kernelSize//2)*2)/stride,
            (inputShape[1]+(kernelSize//2)*2)/stride,
            numberOfKernels
        ] 

        myTransposeConvolutionLayer=TransposeConvolutionLayer()
        myTransposeConvolutionLayer.newLayer(kernelSize,numberOfKernels,stride,layerInputChannels,seed)
        self.layers.append(myTransposeConvolutionLayer)
        self.layerKey.append("TRANSPOSECONVOLUTE")
        self.layerOutputShape.append(outputShape)
        self.totalTrainableVariables+=kernelSize**2*layerInputChannels*numberOfKernels

    def addPoolingLayer(self,size,stride):
        #check that the previous layer is not flat
        if(len(self.layers)!=0):
            if(self.layerOutputShape[-1]==1):
                #the previous layer is flat
                #a convolutional layer can't go here
                raise(invalidLayerPlacement(True,False,True))       

        #grab input shpae
        if(len(self.layers)==0):
            inputShape=[self.inputSize,self.inputSize,self.inputChannels]
        else:
            inputShape=self.layerOutputShape[-1]

        #calculate output shape same way as convolutional layer
        outputShape=[
            (inputShape[0]-(size//2)*2)/stride,
            (inputShape[1]-(size//2)*2)/stride,
            inputShape[2]
        ] 
        
        #if we are here we are good to go
        myPoolingLayer=PoolingLayer()
        myPoolingLayer.newLayer(size,stride)
        self.layers.append(myPoolingLayer)
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

    def addInstanceNormalizationLayer(self,mean,stddev):
        if(len(self.layers)!=0):
            if(self.layerOutputShape[-1]==1):
                #previous layer is flat
                raise(invalidLayerPlacement(True,False,True))
        
        if(len(self.layers)==0):
            inputShape=[self.inputSize,self.inputSize,self.inputChannels]
        else:
            inputShape=self.layerOutputShape[-1]
        
        outputShape=inputShape.copy()

        myInstanceNormalizationLayer=InstanceNormalizationLayer()
        myInstanceNormalizationLayer.newLayer(mean,stddev) 
        self.layers.append(myInstanceNormalizationLayer)
        self.layerKey.append("INSTANCENORMALIZATION")
        self.layerOutputShape.append(outputShape)


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

    #calculates error from set without training
    def validate(self,X,Y):
        guess=self.evaluate(X)
        #calculate error using sqared error
        error=0
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                error+=(guess[i][j]-Y[i][j])**2
        error=error/len(Y)
        return(error)


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
    def exportNetwork(self,path):
        from os import mkdir
        #create directory if one does not already exists
        try:
            mkdir(path)
        except FileExistsError:
            pass
        except Exception as e:
            raise(invalidPath(path))
        if self.debug:
            print("exporing layer makeup to:",path+"\\layers.txt")
        #write layer makeup to text file
        with open(path+"\\layers.txt","w") as f:
            for i in self.layerKey:
                f.write(i+"\n")
            
        #save each layer
        for i,layer in enumerate(self.layers):
            if self.debug:
                print("exporting layer",i+1,"of",len(self.layers),"\t"+path+"\\LAYER"+str(i)+self.layerKey[i])
            if(self.layerKey[i]!="FLATTEN"):
                layer.exportLayer(path,"LAYER"+str(i)+self.layerKey[i])
        


    #import a network of the format given above
    def importNetwork(self,myPath):
        from os import path

        #check if directory exists
        if not path.exists(myPath):
            raise(missingDirectoryForImport(myPath))


        #import layer makeup from file 
        try:
            with open(myPath+"\\layers.txt","r") as f:
                fileLines=f.readlines() 
                #strip line breaks
                fileLines=[i[:-1] for i in fileLines]
            for i,line in enumerate(fileLines):
                if self.debug:
                    print("importing layer",i+1,"of",len(fileLines),"\t",myPath+"\\LAYER"+str(i)+line)
                if line =="CONVOLUTE":
                    myConvolutionLayer=ConvolutionLayer()
                    #throws stuff, passed on
                    myConvolutionLayer.importLayer(myPath,"LAYER"+str(i)+"CONVOLUTE")
                    self.layers.append(myConvolutionLayer)
                    self.layerKey.append(line)
                elif line=="TRANSPOSECONVOLUTE":
                    myTransposeConvolutionLayer=TransposeConvolutionLayer()
                    #throws stuff, passed on
                    myTransposeConvolutionLayer.importLayer(myPath,"LAYER"+str(i)+"TRANSPOSECONVOLUTE")
                    self.layers.append(myTransposeConvolutionLayer)
                    self.layerKey.append(line)
                elif line =="POOL":
                    myPoolingLayer=PoolingLayer()
                    #throws stuff, passed on
                    myPoolingLayer.importLayer(myPath,"LAYER"+str(i)+"POOL")
                    self.layers.append(myPoolingLayer)
                    self.layerKey.append(line) 
                elif line =="FLATTEN":
                    myFlattenLayer=FlattenLayer()
                    #flatten layer does not have to be imported
                    self.layers.append(myFlattenLayer)
                    self.layerKey.append(line) 
                elif line =="DENSE":
                    myDenseLayer=DenseLayer()
                    #throws stuff, passed on
                    myDenseLayer.importLayer(myPath,"LAYER"+str(i)+"DENSE")
                    self.layers.append(myDenseLayer)
                    self.layerKey.append(line) 
                elif line=="INSTANCENORMALIZATION":
                    myInstanceNormalizationLayer=InstanceNormalizationLayer()
                    myInstanceNormalizationLayer.importLayer(myPath,"LAYER"+str(i)+"DENSE")
                    self.layers.append(myInstanceNormalizationLayer)
                    self.layerKey.append(line)
                else:
                    raise(invalidDataInFile(myPath+"\\layers.txt","LAYER"+str(i),line))
        except IOError:
            raise(missingFileForImport(myPath+"\\layers.txt"))

    #return deepcopy of self
    #TODO
    def deepcopy(self):
        return
        #TODO 
