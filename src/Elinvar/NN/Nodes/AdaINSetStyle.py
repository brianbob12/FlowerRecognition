from typing import List
from tensorflow import reduce_mean,transpose,broadcast_to,split
from Elinvar.NN.Nodes.BuildableNode import BuildableNode
from tensorflow.math import reduce_std
from tensorflow import concat,Tensor
from ..Exceptions import *
from .Node import Node

#performs instance normalization
#normalizes values within each channel
#must take inputs of shape [batch,chanells,height,width]

class AdaINSetStyle(BuildableNode):

  def __init__(self,name:Optional[str]=None,protected:Optional[bool]=None,ID:Optional[int]=None):
    super().__init__(name=name,protected=protected,ID=ID)
    self.hasTrainableVariables:bool=False
    self.totalTrainableVariables:int=0
    self.imported:bool=False
    self.inputChannels:int=0

    #variables used to manage connections
    self.inputShape:List[int]=[0,0,0]
    self.connectedChannels:int=0

  #outputs data with specified stddev and mean
  
  #WARNING: the following is complicated and not memory efficient
  def execute(self,inputs:List[Tensor]):
    #check that the inputs meet the required format
    if len(inputs)!=len(self.inputConnections):
      raise(invalidNumberOfNodeInputs(len(inputs),len(self.inputConnections)))

    styleInput=inputs[0]

    #split the style input into mean and stddev
    targetMeans,targetStddevs=split(styleInput,num_or_size_splits=2,axis=-1)

    imageInputs=inputs[1:]
    #concat imageInputs
    inputChannels=concat(imageInputs,-1)
    inputShape=inputChannels.shape

    #normalize inputChannels
    #normalize each channel separately

    means=reduce_mean(inputChannels,[-2,-3])#shape=[batch,channels]
    standardDeviations=reduce_std(inputChannels,[-2,-3])#shape=[batch,channels]
    standardDeviations=standardDeviations/targetStddevs
    standardDeviations=standardDeviations+1e-8#add small value to prevent division by zero
    #NOTE: the following madness creates new memory addresses full of stuff
    #this should be updated at somepoint to make it much more vRAM efficient

    #this gets means ready to do an itemwise subtraction
    #it has to have the same shape as the input
    formattedMeans=transpose(
      broadcast_to(means,[inputShape[1],inputShape[2],inputShape[0],inputShape[3]]),
      perm=[2,0,1,3])
    formattedTargetMeans=transpose(
      broadcast_to(targetMeans,[inputShape[1],inputShape[2],inputShape[0],inputShape[3]]),
      perm=[2,0,1,3])

    #gets it ready for itemwise division
    formattedStddev=transpose(
      broadcast_to(standardDeviations,[inputShape[1],inputShape[2],inputShape[0],inputShape[3]]),
      perm=[2,0,1,3])

    out=((inputChannels-formattedMeans)/formattedStddev)+formattedTargetMeans  

    return out

  def connect(self,connections:List[Node]):
    if len(connections)==0:
      return

    imageConnections:List[Node]=[]
    if len(self.inputConnections)==0:
      #the style input determines means and standard deviations
      self.inputChannels=int(connections[0].outputShape[0]/2)
      #check that there are an even number of floats in the style input
      if connections[0].outputShape[0]%2!=0:
        raise(invalidNodeConnection(connections[0].outputShape,["None*2"]))

      imageConnections=connections[1:]
    else:
      imageConnections=connections    

    #checks
    #if there are image connections already established
    if len(self.inputConnections)>1:
      shape0=self.inputShape[0]
      shape1=self.inputShape[1]      
    else:
      shape0=imageConnections[0].outputShape[0]
      shape1=imageConnections[0].outputShape[1]
      #this check is important
      if shape0 ==None or shape1==None:
        #NOTE: this should really be a different error
        raise(invalidNodeConnection(imageConnections[0].outputShape,[None,None,None]))

    for prospectNode in imageConnections:
      if prospectNode.outputShape[0]!=shape0 or prospectNode.outputShape[1]!=shape1:
        raise(invalidNodeConnection(prospectNode.outputShape,[shape0,shape1,None]))
      self.connectedChannels+=prospectNode.outputShape[2]

    #connect

    self.inputShape=[shape0,shape1,self.inputChannels]
    self.outputShape=self.inputShape#NOTE: same memory address
    super().connect(connections)

  def build(self, seed: Optional[int] = None) -> int:
    #check that the right number of channels are connected
    if self.connectedChannels!=self.inputChannels:
      raise(invalidNumberOfNodeInputs(self.connectedChannels,self.inputChannels))

    return super().build(seed)

  
  def exportNode(self, path, subdir):
      accessPath= super().exportNode(path, subdir)

      #save type
      #NOTE this will be overwritten by children
      #therefore this saves the lowest class of the node
      with open(accessPath+"\\type.txt","w") as f:
        f.write("AdaINSetStyle") 

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