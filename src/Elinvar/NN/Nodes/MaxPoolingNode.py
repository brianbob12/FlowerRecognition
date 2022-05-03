
from os import access
from sqlite3 import connect
from turtle import shape
from tensorflow import concat
from .Node import Node

from tensorflow.nn import max_pool2d
import numpy as np
from ..Exceptions import invalidDataInFile, invalidNodeConnection, invalidPath, missingDirectoryForImport, missingFileForImport

#non-trainable layer#
#maxpooling
class MaxPoolingNode(Node):
  def __init__(self,name=None,protected=False,ID=None):
    super().__init__(name=name,protected=protected,ID=ID)
    self.hasTrainableVariables=False
    self.totalTrainableVariables=0
    self.inputShape=None

  def newLayer(self,size,stride):
    self.size=size
    self.stride=stride
    self.inputChannels=0#until connections are set
    self.padding=0
    
  def execute(self,inputs):
    myInput=concat(inputs,-1)
    return max_pool2d(myInput,[self.size,self.size],self.stride,"VALID")

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
      if shape0==None or shape1==None:
        #NOTE: this should really be a different error
        raise(invalidNodeConnection(connections[0].outputShape,[None,None,None,None]))
    
    for prospectNode in connections:
      if prospectNode.outputShape[0]!=shape0 or prospectNode.outputShape[1]!=shape1:
        raise(invalidNodeConnection(prospectNode.outputShape,[shape0,shape1,None])) 

    #connect
    for node in connections:
      self.inputChannels+=node.outputShape[-1]

    self.inputShape=[shape0,shape1,self.inputChannels]
    self.outputShape=[
      int((self.inputShape[0]-self.size+self.padding*2)/self.stride+1),
      int((self.inputShape[1]-self.size+self.padding*2)/self.stride+1),
      self.inputChannels
    ]

    super().connect(connections)

  def exportNode(self, path, subdir):
      accessPath = super().exportNode(path, subdir)

      #save type
      #NOTE this will be overwritten by children
      #therefore this saves the lowest class of the node
      with open(accessPath+"\\type.txt","w") as f:
        f.write("MaxPoolingNode")

      #save hyper.txt
      #has data:size,stride
      with open(accessPath+"\\hyper.txt","w") as f:
        f.write(str(self.size)+"\n")
        f.write(str(self.stride)+"\n")
        f.write(str(self.padding)+"\n")

      return accessPath

  def importNode(self, myPath, subdir):
      accessPath,connections = super().importNode(myPath, subdir)

      #import from hyper.txt
      try:
        with open(accessPath+"\\hyper.txt","r") as f:
          fileLines=f.readlines()
          #strip line breaks
          fileLines=[i[:-1] for i in fileLines]
          try:
            self.size=int(fileLines[0])
          except ValueError as e:
            raise(invalidDataInFile(accessPath+"\\hyper.txt","size",fileLines[0]))
          try:
            self.stride=int(fileLines[1])
          except ValueError as e:
            raise(invalidDataInFile(accessPath+"\\hyper.txt","stride",fileLines[1]))
          try:
            self.padding=int(fileLines[2])
          except ValueError as e:
            raise(invalidDataInFile(accessPath+"\\hyper.txt","padding",fileLines[2]))
      except IOError:
        raise(missingFileForImport(accessPath+"\\hyper.txt"))

      self.inputChannels=0

      return accessPath,connections
    
