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

from typing import Callable, Dict, List
from tensorflow import (Variable,matmul,constant)
from .BuildableNode import BuildableNode
from tensorflow.random import normal
from tensorflow.nn import relu,sigmoid
from tensorflow.math import tanh
from tensorflow import Tensor

from ..Exceptions import *

class DenseLayer(BuildableNode):
  def __init__(self,name:Optional[str]=None,protected:bool=False,ID:Optional[int]=None):
    super().__init__(name=name,protected=protected,ID=ID)
    self.hasTrainableVariables=True 
    self.activationLookup:Dict[str,Callable[[Tensor],Tensor]]={"relu":relu,"linear":self.linear,"sigmoid":sigmoid,"tanh":tanh}
  
  def linear(self,x:Tensor)->Tensor:
    return (x)

  #throws unknownActivationFunction if activation function not it activationLookup
  #TODO replace activationFunction with class or Enum
  def newLayer(self,layerSize:int,activationFunction:str):
    self.size=layerSize
    self.outputShape:List[int]=[layerSize]
    self.activationKey:str=activationFunction
    try:
      self.activation=self.activationLookup[activationFunction]
    except KeyError as e:
      raise unknownActivationFunction(activationFunction)
 


  #TODO make it throw an error if inputShape not [None]
  def build(self,seed=None) -> int:
    if self.built:
      self.totalTrainableVariables

    if len(self.inputConnections)<1:
      raise(notEnoughNodeConnections(len(self.inputConnections),1))
    #now make the variables

    inputShape=self.inputConnections[0].outputShape
    self.inputSize=inputShape[0]

    biasInit=0.1
    weightInitSTDDEV=1/self.inputSize

    self.biases=Variable(constant(biasInit,shape=[self.size]))
    self.weights=Variable(normal([self.inputSize,self.size],stddev=weightInitSTDDEV,mean=0,seed=seed))

    self.built=True

    self.totalTrainableVariables=self.inputSize*self.size
    return self.totalTrainableVariables

 
  #function that executes the layer for a list of inputs
  #inp has shape [None,inputSize]
  #returns shape [None,outputSize] 
  def execute(self,inp):
    if not self.built:
      raise(operationWithUnbuiltNode(self.ID,"execute"))
    else:
      return((self.activation(matmul([inp],self.weights)+self.biases))[0])

  #function that returns a shape [2] list of trainable variables
  #because these are tf varialbe it is returning a list of pointers
  def getTrainableVariables(self):
    #does error checking
    super().getTrainableVariables()
    #the set of weights and the biases are each a single multi-dimensional variable
    return([self.biases,self.weights])

  def connect(self, connections):
    if len(connections)==0: return
    if len(connections[0].outputShape)!=1:
      raise(invalidNodeConnection(connections[0].outputShape,[None]))
    super().connect(connections)

  #Creates a directory for the layer

  #export to a weight and bias file
  #files:
  # [path]/[subdir]/mat.weights (byteformat)
  # [path]/[subdir]/mat.biases (byteformat)
  # [path]/[subdir]/hyper.txt
  def exportNode(self,path,subdir):
    import struct

    accessPath=super().exportNode(path,subdir)

    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("DenseLayer")

    #save hyper.txt
    #contins: inputSize, layerSize, activation 
    with open(accessPath+"\\hyper.txt","w") as f:
      f.write(str(self.inputSize)+"\n")
      f.write(str(self.size)+"\n") 
      f.write(self.activationKey+"\n")
    
    #save mat.weights
    weightFloats=[]
    for i in range(self.weights.get_shape()[0]):
      for j in range(self.weights[i].get_shape()[0]):
        weightFloats.append(float(self.weights[i][j]))
    with open(accessPath+"\\mat.weights","wb") as f:      
      f.write(bytearray(struct.pack(str(len(weightFloats))+"f",*weightFloats)))

    del weightFloats#this is important because this can be very large and the function can take a long time to load

    #save mat.biases
    biasFloats=[]
    with open(accessPath+"\\mat.biases","wb") as f:
      for i in range(self.biases.get_shape()[0]):
        biasFloats.append(float(self.biases[i]))
      f.write(bytearray(struct.pack(str(len(biasFloats))+"f",*biasFloats)))

    return accessPath


  #function that loads a layer from files and stores perameters on stack
  #gets from to [path]/[subdir]
  def importNode(self,myPath,subdir):
    

    from os import path

    accessPath,connections=super().importNode(myPath,subdir)

    #import from hyper.txt
    try:
      with open(accessPath+"\\hyper.txt","r") as f:
        fileLines=f.readlines()
        #strip line breaks
        fileLines=[i[:-1] for i in fileLines]
        try:
          self.inputSize=int(fileLines[0])
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","inputSize",fileLines[0]))
        try:
          self.size=int(fileLines[1]) 
          self.outputShape=[self.size]
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","size",fileLines[1]))
        try:
          self.activationKey=fileLines[2]
          self.activation=self.activationLookup[fileLines[2]]
        except KeyError as e:
          raise unknownActivationFunction(fileLines[2])
    except IOError:
      raise(missingFileForImport(accessPath,"hyper.txt"))

    
    #import weights
    import struct

    try:
      with open(accessPath+"//mat.weights","rb") as f:
        raw=f.read()#type of bytes

        try:
          inp=struct.unpack(str(self.inputSize*self.size)+"f",raw)#list of float32s
        except struct.error as e:
          raise(invalidByteFile(accessPath+"//mat.weights"))
        
        weights=[] 
        for i in range(self.inputSize):
          weights.append([])
          for j in range(self.size):
            weights[i].append(inp[i*self.size+j])
        self.weights=Variable(weights)

    except IOError:
      raise(missingFileForImport(accessPath,"mat.weights"))

    #import biases
    try:
      with open(accessPath+"//mat.biases","rb") as f:
        raw=f.read()#type of bytes
        try:
          inp=struct.unpack(str(self.size)+"f",raw)#list of float32s
        except struct.error as e:
          raise(invalidByteFile(accessPath+"//mat.biases"))
        self.biases=Variable([i for i in inp])

    except IOError:
      raise(missingFileForImport(accessPath,"mat.biases"))

    self.built=True

    return accessPath,connections

     
