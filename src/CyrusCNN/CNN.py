
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
    def export(self,path):#throws IOerror
        import os
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        except Exception as e:
            raise(badPath(path))



        #store hyperperanmeters in hyper.txt
        with open(path+"\\hyper.txt","w") as f:
            f.write(str(self.inputSize)+"\n")
            f.write(str(self.outputSize)+"\n")
            for i,v in enumerate(self.nHidden):
                f.write(str(v))
                if(i+1!=len(self.nHidden)):
                    f.write(",")
            f.write("\n")
            for i,v in enumerate(self.activation):
                f.write(v)
                if(i+1!=len(self.activation)):
                    f.write(",")
            f.write("\n")


        #set weight and bias files as comma sperated values
        #note: some percision will be lost when converting binary floats to strings
        for i in range(len(self.nHidden)+1):
            out=[]
            with open(path+"\\w"+str(i)+".weights","wb") as f:
                for j in range(self.weights[i].get_shape()[0]):
                    for k in range(self.weights[i][j].get_shape()[0]):
                        out.append(float(self.weights[i][j][k]))
                f.write(bytearray(struct.pack(str(len(out))+"f",*out)))
            out=[]
            with open(path+"\\b"+str(i)+".biases","wb") as f:
                for j in range(self.biases[i].get_shape()[0]):
                    out.append(float(self.biases[i][j]))
                f.write(bytearray(struct.pack(str(len(out))+"f",*out)))

    #import a network of the format given above
    def importNetwork(self,path):#throws IOerror and byte error
        #check if the path is real
        import os
        if(not os.path.exists(path)):
            raise(badPath(path))

        #check hyperperameters
        try:
            with open(path+"\\hyper.txt","r") as f:
                hyperPerameters=f.readlines()
        except IOError:
            raise(missingFile(path+"\\hyper.txt"))
        try:
            self.inputSize=int(hyperPerameters[0])
            self.outputSize=int(hyperPerameters[1])
            self.nHidden=[int(i) for i in hyperPerameters[2].split(",")]
            self.activation=hyperPerameters[3][:-1].split(",")#exclude final \n
        except:
            raise(fileMissingData(path+"\\hyper.txt"))

        #pre-network safety checks(passing erros)
        if(len(self.activation)!=len(self.nHidden)+1):
            raise(unspecifiedActivation)#see Network.Exceptions
        for i in self.activation:
            if not i in self.activationLookup.keys():
                raise(unknownActivationFunction(i))

        #initalise variables
        self.biases=[]
        self.weights=[]
        #I know this is a little messy. It is the same segment three times
        try:
            with open(path+"\\w0.weights","rb") as f:
                raw=f.read()#type of bytes
                inp=struct.unpack(str(self.inputSize*self.nHidden[0])+"f",raw)#list of int
                tad=[]
                for j in range(self.inputSize):
                    tad2=[]
                    for k in range(self.nHidden[0]):
                        tad2.append(inp[j*self.nHidden[0]+k])
                    tad.append(tad2)
                self.weights.append(Variable(tad))
        except IOError:
            raise(missingFile(path,path+"\\w0.weights"))
        try:
            with open(path+"\\b0.biases","rb") as f:
                raw=f.read()#type of bytes
                inp=struct.unpack(str(self.nHidden[0])+"f",raw)#list of floats
                self.biases.append(Variable([i for i in inp]))

        except IOError:
            raise(missingFile(path,path+"\\b0.bases"))

        for i in range(1,len(self.nHidden)):
            try:
                with open(path+"\\w"+str(i)+".weights","rb") as f:
                    raw=f.read()#type of bytes
                    #print(path+"\\w"+str(i)+".weights")#DEBUG
                    inp=struct.unpack(str(self.nHidden[i-1]*self.nHidden[i])+"f",raw)#list of int
                    tad=[]
                    for j in range(self.nHidden[i-1]):
                        tad2=[]
                        for k in range(self.nHidden[i]):
                            tad2.append(inp[j*self.nHidden[i]+k])
                        tad.append(tad2)
                    self.weights.append(Variable(tad))

            except IOError:
                raise(missingFile(path,path+"\\w"+str(i)+".weights"))

            try:
                with open(path+"\\b"+str(i)+".biases","rb") as f:
                    raw=f.read()#type of bytes
                    inp=struct.unpack(str(self.nHidden[i])+"f",raw)#list of int
                    self.biases.append(Variable([i for i in inp]))

            except IOError:
                raise(missingFile(path,path+"\\b"+str(i)+".bases"))

        try:
            with open(path+"\\w"+str(len(self.nHidden))+".weights","rb") as f:
                raw=f.read()#type of bytes
                inp=struct.unpack(str(self.nHidden[-1]*self.outputSize)+"f",raw)#list of int
                tad=[]
                for j in range(self.nHidden[-1]):
                    tad2=[]
                    for k in range(self.outputSize):
                        tad2.append(inp[j*self.outputSize+k])
                    tad.append(tad2)
                self.weights.append(Variable(tad))

        except IOError:
            raise(missingFile(path,path+"\\w"+str(len(self.nHidden))+".weights"))
        try:
            with open(path+"\\b"+str(len(self.nHidden))+".biases","rb") as f:
                raw=f.read()#type of bytes
                inp=struct.unpack(str(self.outputSize)+"f",raw)#list of int
                self.biases.append(Variable([i for i in inp]))

        except IOError:
            raise(missingFile(path,path+"\\b"+str(len(self.nHidden))+".bases"))

    #return deepcopy of self
    def deepcopy(self):
        return
        #todo 
