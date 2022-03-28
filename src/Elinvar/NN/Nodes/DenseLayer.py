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
from .BuildableNode import BuildableNode
from tensorflow.random import normal
from tensorflow.nn import relu,sigmoid
from tensorflow.math import tanh

from ..Exceptions import *

class DenseLayer(BuildableNode):
  def __init__(self,name=None,protected=False):
    super().__init__(name=name,protected=protected)
    self.hasTrainableVariables=True 
    self.activationLookup={"relu":relu,"linear":self.linear,"sigmoid":sigmoid,"tanh":tanh}
  
  def linear(self,x):
    return (x)

  #throws unknownActivationFunction if activation function not it activationLookup
  def newLayer(self,layerSize,activationFunction):
    self.size=layerSize
    self.outputShape=[layerSize]
    self.activationKey=activationFunction
    try:
      self.activation=self.activationLookup[activationFunction]
    except KeyError as e:
      raise unknownActivationFunction(activationFunction)
 


  #TODO make it throw an error if inputShape not [None]
  def build(self,seed=None):
    if self.built:return

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

 
  #function that executes the layer for a list of inputs
  #inp has shape [None,inputSize]
  #returns shape [None,outputSize] 
  def execute(self,inp):
    if not self.built:
      raise(operationWithUnbuiltNode("execute"))
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
  def exportLayer(self,path,subdir):
    if not self.built:
      raise(operationWithUnbuiltNode("exportLayer"))

    import struct
    from os import mkdir 

    accessPath=path+"\\"+subdir

    #first step is to create a directory for the network if one does not already exist
    try:
      mkdir(accessPath)
    except FileExistsError:
      pass
    except Exception as e:
      raise(invalidPath(accessPath))

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


  #function that loads a layer from files and stores perameters on stack
  #gets from to [path]/[subdir]
  def importLayer(self,superdir,subdir):
    

    from os import path

    accessPath=superdir+"\\"+subdir
    
    #check if directory exists
    if not path.exists(accessPath):
      raise(missingDirectoryForImport(accessPath))

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
      raise(missingFileForImport(accessPath+"\\hyper.txt"))

    
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

     
