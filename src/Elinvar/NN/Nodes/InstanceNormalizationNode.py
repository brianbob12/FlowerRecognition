
from typing import List
from tensorflow import reduce_mean,transpose,broadcast_to
from tensorflow.math import reduce_std
from tensorflow import concat
from ..Exceptions import *
from .Node import Node

#performs instance normalization
#normalizes values within each channel
#must take inputs of shape [batch,chanells,height,width]

class InstanceNormalizationNode(Node):

  def __init__(self,name=None,protected=None,ID=None):
    super().__init__(name=name,protected=protected,ID=ID)
    self.hasTrainableVariables=False
    self.totalTrainableVariables=0
    self.imported=False
    self.inputChannels=0
    self.inputShape:List[int]=[]

  def newLayer(self,mean,stddev):
    self.stddev=stddev
    self.mean=mean  

  #outputs data with specified stddev and mean
  
  #WARNING: the following is complicated and not memory efficient
  def execute(self,inputs):
    myInputs=concat(inputs,-1)
    inputShape=myInputs.shape
    means=reduce_mean(myInputs,[-2,-3])
    standardDeviations=reduce_std(myInputs,[-2,-3])
    #NOTE: the following madness creates new memory addresses full of stuff
    #this should be updated at somepoint to make it much more vRAM efficient

    #this gets means ready to do an itemwise subtraction
    #it has to have the same shape as the input
    formattedMeans=transpose(
      broadcast_to(means,[inputShape[1],inputShape[2],inputShape[0],inputShape[3]]),
      perm=[2,0,1,3])
    #gets it ready for itemwise division
    formattedStddev=transpose(
      broadcast_to(standardDeviations,[inputShape[1],inputShape[2],inputShape[0],inputShape[3]]),
      perm=[2,0,1,3])

    out=((myInputs-formattedMeans)/formattedStddev)*self.stddev + self.mean
    return out

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
        self.inputChannels+=node.outputShape[2]

    self.inputShape=[shape0,shape1,self.inputChannels]
    self.outputShape=self.inputShape#NOTE: same memory adress
    super().connect(connections)
  
  def exportNode(self, path, subdir):
      accessPath= super().exportNode(path, subdir)

      #save type
      #NOTE this will be overwritten by children
      #therefore this saves the lowest class of the node
      with open(accessPath+"\\type.txt","w") as f:
        f.write("InstanceNormalizationNode") 

      #save hyper.txt
      #has data:size,stride
      with open(accessPath+"\\hyper.txt","w") as f:
        f.write(str(self.mean)+"\n")
        f.write(str(self.stddev)+"\n")

      return accessPath

  def importNode(self, myPath, subdir):
      accessPath,connections= super().importNode(myPath, subdir)

      #import from hyper.txt
      try:
        with open(accessPath+"\\hyper.txt","r") as f:
          fileLines=f.readlines()
          #strip line breaks
          fileLines=[i[:-1] for i in fileLines]
          try:
            self.mean=int(fileLines[0])
          except ValueError as e:
            raise(invalidDataInFile(accessPath+"\\hyper.txt","mean",fileLines[0]))
          try:
            self.stddev=int(fileLines[1])
          except ValueError as e:
            raise(invalidDataInFile(accessPath+"\\hyper.txt","stddev",fileLines[1]))
      except IOError:
        raise(missingFileForImport(accessPath,"hyper.txt"))
      self.imported=True
      return accessPath,connections 