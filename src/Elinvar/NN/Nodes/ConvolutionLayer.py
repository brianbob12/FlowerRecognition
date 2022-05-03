from multiprocessing import connection
import numpy as np
from tensorflow import Variable,concat
from tensorflow.nn import conv2d
from tensorflow.random import normal
from .BuildableNode import BuildableNode

from ..Exceptions import *

class ConvolutionLayer(BuildableNode):
  
  def __init__(self,name=None,protected=False,ID=None):
    super().__init__(name=name,protected=protected,ID=ID)
    self.hasTrainableVariables=True
    #tracks if network has been imported
    self.imported=False

    #filterSize is interger
    #stride is a single int
  def newLayer(self,kernelSize,numberOfKernels,stride,padding):
    weightInitSTDDEV=0.1
    self.kernelSize=kernelSize
    self.padding=padding
    self.numberOfKernels=numberOfKernels
    self.inputChannels=0#until connections are set
    self.imported=False
    self.strideSize=stride 
    self.strides=[1,stride,stride,1]
    self.inputShape=None
  
  def build(self,seed=None) -> int:
    if self.built: return

    if len(self.inputConnections)<1:
      raise(notEnoughNodeConnections(len(self.inputConnections),1)) 


    self.filter=Variable(normal(shape=[self.kernelSize,self.kernelSize,self.inputChannels,self.numberOfKernels],seed=seed))
    self.built=True
    self.totalTrainableVariables=self.kernelSize*self.kernelSize*self.inputChannels*self.numberOfKernels
    return self.totalTrainableVariables
     

  def connect(self,connections):
    if len(connections)==0:
      return
    #checks
    if len(self.inputConnections)>0:
      shape0=self.inputShape[0]
      shape1=self.inputShape[1]      
    else:
      shape0=connections[0].outputShape[0]
      shape1=connections[0].outputShape[1]
      #this check is important
      if shape0 ==None or shape1==None:
        #NOTE: this should really be a different error
        raise(invalidNodeConnection(connections[0].outputShape,[None,None,None]))

    for prospectNode in connections:
      if prospectNode.outputShape[0]!=shape0 or prospectNode.outputShape[1]!=shape1:
        raise(invalidNodeConnection(prospectNode.outputShape,[shape0,shape1,None]))

    #connect
    if not self.imported:
      for node in connections:
        self.inputChannels+=node.outputShape[-1]

    self.inputShape=[shape0,shape1,self.inputChannels]
    self.outputShape=[
      int((shape0-self.kernelSize+self.padding*2)/self.strideSize+1),
      int((shape1-self.kernelSize+self.padding*2)/self.strideSize+1),
      self.numberOfKernels
    ]
    super().connect(connections)

  #inputs have shape [None,a,a,3] tf.float32
  def execute(self,inputs):
    if not self.built:
      raise(operationWithUnbuiltNode("execute"))
    else:
      myInput=concat(inputs,-1)
      return conv2d(myInput,self.filter,self.strides,"VALID")

  #return a list of the trainable variables
  def getTrainableVariables(self):
    #does error checking
    super().getTrainableVariables()
    return [self.filter]

  def exportNode(self,path,subdir):

    import struct
    
    accessPath=super().exportNode(path,subdir)

    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("ConvolutionLayer")

 

    #save hyper.txt
    #contains: fliterSize, strides 
    with open(accessPath+"\\hyper.txt","w") as f:
      f.write(str(self.kernelSize)+"\n")
      f.write(str(self.strideSize)+"\n") 
      f.write(str(self.numberOfKernels)+"\n")
      f.write(str(self.inputChannels)+"\n")
      f.write(str(self.padding)+"\n")
    
    #save mat.kernel
    kernelFloats=[]
    with open(accessPath+"\\mat.filter","wb") as f:
      for i in range(self.kernelSize):
        for j in range(self.kernelSize):
          for k in range(self.inputChannels):
            for l in range(self.numberOfKernels):
              kernelFloats.append(float(self.filter[i][j][k][l]))
      f.write(bytearray(struct.pack(str(len(kernelFloats))+"f",*kernelFloats)))
    return accessPath

  def importNode(self,myPath,subdir):

    accessPath,connections=super().importNode(myPath,subdir)
  

    #import from hyper.txt
    try:
      with open(accessPath+"\\hyper.txt","r") as f:
        fileLines=f.readlines()
        #strip line breaks
        fileLines=[i[:-1] for i in fileLines]
        try:
          self.kernelSize=int(fileLines[0])
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","kernelSize",fileLines[0]))
        try:
          self.strideSize=int(fileLines[1]) 
          self.strides=[1,self.strideSize,self.strideSize,1]
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","size",fileLines[1])) 
        try:
          self.numberOfKernels=int(fileLines[2])
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","filterSize",fileLines[2]))
        try:
          self.inputChannels=int(fileLines[3]) 
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","size",fileLines[3])) 
        try:
          self.padding=int(fileLines[4]) 
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\hyper.txt","padding",fileLines[4])) 
    except IOError:
      raise(missingFileForImport(accessPath+"\\hyper.txt"))

  
    #import kernel
    import struct

    try:
      with open(accessPath+"\\mat.filter","rb") as f:
        raw=f.read()#type of bytes  
        try:
          inp=struct.unpack(str(self.kernelSize*self.kernelSize*self.inputChannels*self.numberOfKernels)+"f",raw)#list of float32s
        except struct.error as e:
          raise(invalidByteFile(accessPath+"\\mat.filter"))
      
        kernel=[] 
        try:
          for i in range(self.kernelSize):
            kernel.append([])
            for j in range(self.kernelSize):
              kernel[i].append([])
              for k in range(self.inputChannels):
                kernel[i][j].append([])
                for l in range(self.numberOfKernels):
                  #TODO check this is correct
                  kernel[i][j][k].append(
                    inp[i*self.kernelSize*self.inputChannels*self.numberOfKernels+j*self.inputChannels*self.numberOfKernels+k*self.numberOfKernels+l])
        except Exception as e:
          raise(invalidByteFile(accessPath+"//mat.filer"))

        self.filter=Variable(kernel)

    except IOError:
      raise(missingFileForImport(accessPath,"mat.filter"))
    self.imported=True
    self.built=True

    return accessPath,connections