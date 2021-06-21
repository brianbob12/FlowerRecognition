#dense layer of nerual network
#has option for actication funtions

#functionality:
#new layer
#import layer from file(s)
#export layer to file(s)
#(eager) execute layer
#return trainable varialbes for training
#maybe deepcopy


#dependencies

from tensorflow import (Variable,matmul,constant)
from CyrusCNN.Exceptions import unknownActivationFunction
from tensorflow.random import truncated_normal
from tensorflow.nn import relu,sigmoid
from tensorflow.math import tanh

from .Exceptions import *

class DenseLayer():
  def __init__(self):
    #initalise a map of string to function for activation fuctions
    self.activationLookup={"relu":relu,"linear":self.linear,"sigmoid":sigmoid,"tanh":tanh}
  
  def linear(self,x):
    return (x)

  #create a new layer form randomly initialized values
  def newLayer(self,inputSize,layerSize,activation):
    self.inputSize=inputSize
    self.size=layerSize

    try:
      self.activation=self.activationLookup[activation]
    except KeyError as e:
      raise unknownActivationFunction(activation)
    
    #now make the variables

    biasInit=0.1
    weightInitSTDDEV=0.1

    self.biases=Variable(constant(biasInit,shape=[layerSize]))
    self.weights=Variable(truncated_normal([inputSize,layerSize],stddev=weightInitSTDDEV))
 
  #function that executes the layer for a list of inputs
  #inp has shape [None,inputSize]
  #returns shape [None,outputSize] 
  def execute(self,inp):
    return((self.activation(matmul([inp],self.weights)+self.biases))[0])

  #function that returns a shape [2] list of trainable variables
  #because these are tf varialbe it is returning a list of pointers
  def getTrainableVariables(self):
    #the set of weights and the biases are each a single multi-dimensional variable
    return([self.biases,self.weights])

  #export to a weight and bias file
  #saves to [path]/[prefix].weights and [path]/[prefix].biases
  #this function does not check if the file exsists
  def export(self,path,prefix):
    import struct

    out=[]

    with open(path+"\\"+prefix+".weights","wb") as f:
      for i in range(self.weights.get_shape()[0]):
        for j in range(self.weights[i].get_shape()[0]):
          out.append(float(self.weights[i][j]))
      f.write(bytearray(struct.pack(str(len(out))+"f",*out)))

    out=[]

    with open(path+"\\"+prefix+".biases","wb") as f:
      for i in range(self.biases.get_shape()[0]):
        out.append(float(self.biases[i]))
      f.write(bytearray(struct.pack(str(len(out))+"f",*out)))

#function that loads a layer from files
#gets from to [path]/[prefix].weights and [path]/[prefix].biases
#this function does not check if the file exsists
def importLayer(self,path,prefix,inputSize,layerSize,activation):

  self.inputSize=inputSize
  self.size=layerSize

  try:
    self.activation=self.activationLookup[activation]
  except KeyError as e:
    raise unknownActivationFunction(activation)

  import struct

  try:
    with open(path+"\\"+prefix+".weights","rb") as f:
      raw=f.read()#type of bytes
      inp=struct.unpack(str(inputSize*layerSize)+"f",raw)#list of float32s
      tad=[]
      for j in range(inputSize):
        tad2=[]
        for k in range(layerSize):
          tad2.append(inp[j*layerSize+k])
        tad.append(tad2)
        self.weights=Variable(tad)

  #error handeling
  except IOError:
    raise(missingFile(path,path+"\\"+prefix+".weights"))

  try:
    with open(path+"\\"+prefix+".biases","rb") as f:
      raw=f.read()#type of bytes
      inp=struct.unpack(str(layerSize)+"f",raw)#list of float32s
      self.biases=Variable([i for i in inp])

  except IOError:
    raise(missingFile(path,path+"\\"+prefix+".bases"))

     
