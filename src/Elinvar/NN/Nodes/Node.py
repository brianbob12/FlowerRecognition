
#class that holds nodes in the network
#these nodes may be junctions, layers, inputs or outputs
#each node has input connections to other nodes each connection has a shape
from abc import ABC, abstractmethod
from tensorflow import Variable,Tensor
from typing import  Any, List, Tuple
from ..Exceptions import *
#each node has a stored value per execution
#the node can be cleared to clear this value thorugh the function clear
#a node can be perserved which disables the clear function

class Node(ABC):
  __slots__=(
    "name",
    "inputConnections",
    "outputShape",
    "value",
    "executed",
    "protected",
    "hasTrainableVariables",
    "ID",
    "totalTrainableVariables"
  )

  def __init__(self,name:Optional[str]=None,protected:bool=False,ID:Optional[int]=None):
    self.name:Optional[str]=name
    self.inputConnections:List[Any]=[]#list of nodes
    self.outputShape:List[int]=[]#outputshape
    self.executed:bool=False
    self.protected:bool=protected
    self.value:Optional[Tensor]=None
    self.hasTrainableVariables:bool=False
    self.totalTrainableVariables:int=0
    if ID!=None:
      self.ID:int=ID
    else:
      from random import randint
      self.ID:int=randint(0,2**31)

  @abstractmethod
  def execute(self,inputs:List[Tensor])->Tensor:
    pass

  def trainingExecute(self,inputs:List[Tensor])->Tensor:
    return self.execute(inputs)

  def getValue(self)->Tensor:
    if self.value!=None:
      return self.value 
    #else

    #get input values and pass to execute function which will be overridden
    myInputs=[]
    for node in self.inputConnections:
      myInputs.append(node.getValue())

    self.value=self.execute(myInputs)
    self.executed=True
    return self.value

  def getValueTraining(self)->Tensor:
    if self.value !=None:
      return self.value 
    #else

    #get input values and pass to execute function which will be overridden
    myInputs=[]
    for node in self.inputConnections:
      myInputs.append(node.getValueTraining())

    self.value=self.trainingExecute(myInputs)
    self.executed=True
    return self.value

  def clear(self):
    if not self.protected:
      self.value=None
      self.executed=False

  def protectedClear(self):
    self.value=None
    self.executed=False

  def getTrainableVariables(self)->List[Variable]:
    return []
    
  def connect(self,connections:List[Any])->None:
    #Can't specify List[Node] because Node is not yet created
    self.inputConnections=connections
    return

  #this has to be abstract because the type of the node has to be saved correctly
  @abstractmethod
  def exportNode(self,path:str,subdir:str) -> str:
    from os import mkdir 
    import struct

    accessPath=path+"\\"+subdir

    #first step is to create a directory for the network if one does not already exist
    try:
      mkdir(accessPath)
    except FileExistsError:
      pass
    except Exception as e:
      raise(invalidPath(accessPath))

    #save ID.dat
    with open(accessPath+"\\Node.dat","wb") as f:
      f.write(bytearray(struct.pack("i",self.ID)))

    #save connections.txt
    with open(accessPath+"\\connections.txt","w") as f:
      for connection in self.inputConnections:
        f.write(str(connection.ID)+"\n")

    #save type
    #NOTE this will be overwritten by children
    #therefore this saves the lowest class of the node
    with open(accessPath+"\\type.txt","w") as f:
      f.write("Node")

    #write outputShape
    shapeToWrite=""
    for i in self.outputShape:
      shapeToWrite+=str(i)+"\n"
    shapeToWrite=shapeToWrite[:-1]#remove last \n

    with open(accessPath+"\\shape.txt","w") as f:
      f.write(shapeToWrite)

    return(accessPath)
        
  def importNode(self,myPath: str,subdir:str)->Tuple[str,List[int]]:
    from os import path
    import struct

    accessPath=myPath+"\\"+subdir

    #check if directory exists
    if not path.exists(accessPath):
      raise(missingDirectoryForImport(accessPath))

    #read from id from 
    with open(accessPath+"\\Node.dat","rb") as f:
      raw=f.read()

      try:
        inp=struct.unpack("i",raw)
      except struct.error as e:
        raise(invalidByteFile(accessPath+"//Node.dat"))
    self.ID=inp[0]

    #read from connections.txt
    connectionsToResolve:List[int]=[]
    with open(accessPath+"\\connections.txt") as f:
      inp=f.readlines()
      for line in inp:
        try:
          x=int(line)
          connectionsToResolve.append(x)
        except ValueError as e:
          raise(invalidDataInFile(accessPath+"\\connections.txt","connections",line))

    try:
      with open(accessPath+"\\shape.txt","r") as f:
        raw=f.readlines()
    except FileNotFoundError as e:
      raise(missingFileForImport(accessPath,"shape.txt"))
    
    shape=[]
    for i,v in enumerate(raw):
      try:
        shape.append(int(i)) 
      except ValueError as e:
        #because python is interpreted i should contain the last used value of the index
        raise(invalidDataInFile(accessPath+"\\shape.txt",f"shape[{i}]",v))
    self.outputShape=shape

    return(accessPath,connectionsToResolve)

    
