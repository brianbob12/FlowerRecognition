
#class that holds nodes in the network
#these nodes may be junctions, layers, inputs or outputs
#each node has input connections to other nodes each connection has a shape
from os import access
from ..Exceptions import *
#each node has a stored value per execution
#the node can be cleared to clear this value thorugh the function clear
#a node can be perserved which disables the clear function

class Node:
  def __init__(self,name=None,protected=False,ID=None):
    self.name=name
    self.inputConnections=[]#list of nodes
    self.outputShape=None#outputshape
    self.executionValue=None
    self.executed=False
    self.protected=protected
    self.value=None
    self.hasTrainableVariables=False
    if id!=None:
      self.ID=ID
    else:
      from random import randint
      self.ID=randint(0,2**31)

  #private
  #this is made to be overwritten
  def execute(self,inputs):
    return None

  def getValue(self):
    if self.executed:
      return self.value
    #else

    #get input values and pass to execute function which will be overridden
    myInputs=[]
    for node in self.inputConnections:
      myInputs.append(node.getValue())

    self.value=self.execute(myInputs)
    self.executed=True
    return self.value

  def clear(self):
    if not self.protected:
      self.value=None
      self.executed=False

  def protectedClear(self):
    self.value=None
    self.executed=None

  def getTrainableVariables(self):
    return []
    
  def connect(self,connections):
    self.inputConnections=connections
    return

  def exportNode(self,path,subdir):
    from os import mkdir 
    from struct import pack 

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
      f.write(bytearray(pack("i",self.ID)))

    #save connections.txt
    with open(accessPath+"\\connections.txt","w") as f:
      for connection in self.inputConnections:
        f.write(str(connection.ID)+"\n")

    return(accessPath)
        
  def importNode(self,myPath,subdir):
    from os import path
    from struct import unpack

    accessPath=path+"\\"+subdir

    #check if directory exists
    if not path.exists(accessPath):
      raise(missingDirectoryForImport(accessPath))

    return(accessPath)

    
