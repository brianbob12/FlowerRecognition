from multiprocessing.sharedctypes import Value
from random import randint

from Elinvar.NN.Nodes.NodeNameLookup import NodeNameLookup
from .Nodes import *
from tensorflow import GradientTape

from .Exceptions import *

class Network: 
  
  def __init__(self):
    #public
    self.nodes={}
    self.inputNodes=[]
    self.outputNodes=[]
    self.built=False

    #private


  def addNodes(self,nodes):
    for node in nodes:
      #check for ID collisions
      nodeID=node.ID
      changed=False
      while nodeID in self.nodes.keys():
        #check if the node is the same
        if node==self.nodes[nodeID]:
          break
        #else
        nodeID=randint(0,2**31)
        changed=True
      if changed:
        node.ID=nodeID
      self.nodes[nodeID]=node

  def addInputNodes(self,nodes):
    self.addNodes(nodes)
    self.inputNodes+=nodes

  def addOutputNodes(self,nodes):
    self.addNodes(nodes)
    self.outputNodes+=nodes

  def build(self,seed=None):
    #recursive function
    def buildNode(node,seed):
        #first build connections
        for i,inputNode in enumerate(node.inputConnections):
          buildNode(inputNode,seed=seed+3*i)

        #then if buildable, build
        if isinstance(node,BuildableNode):
          if node.built:
            return
          else:
            node.build(seed=seed)

    if seed==None:
      seed=randint(0,2**31)
    for i,outputNode in enumerate(self.outputNodes):
      buildNode(outputNode,seed+i*2)
    self.built=True
    return seed

  def getTotalTrainableVarialbes(self)->int:
    out=0
    for node in self.nodes.values():
      if(node.totalTrainableVariables!=None):
        out+=node.totalTrainableVariables
    return out

  def execute(self,inputs,requestedOutputs):
    #clear
    self.clear()
    #deal with inputs
    for key in inputs.keys():
      self.nodes[key].onExecute=(lambda value:(lambda :value))(inputs[key])
    
    #produce outputs
    return([i.getValue() for i in requestedOutputs])

  def getError(self,inputs,trainingProtocol,getErrorArgs):
    networkOutputs=self.execute(inputs,trainingProtocol.requiredOutputNodes)
    error=trainingProtocol.getError(networkOutputs,getErrorArgs)
    return error

  def train(self,inputs,trainingProtocol,getErrorArgs):
    #all computation has to occur after watching trainableVariables
    #therefore we need to do a clear
    #nodes that are protected may mess this up
    self.clear()

    #get trainable variables
    myTrainableVariables=[]
    for ID,node in self.nodes.items():
      if node.hasTrainableVariables:
        try:
          nodeTrainableVariables=node.getTrainableVariables()
          myTrainableVariables+=nodeTrainableVariables
        except operationWithUnbuiltNode as e:
          if e.operation=="getTrainableVariables":
            pass
          else:
            raise(e)
        except Exception as e:
          raise(e)

    with GradientTape() as g:
      g.watch(myTrainableVariables)
      error=self.getError(inputs,trainingProtocol,getErrorArgs)
      grads=g.gradient(error,myTrainableVariables)

    #TODO move this to TrainingProtocol.py
    optimizer=trainingProtocol.optimizer(trainingProtocol.learningRate)

    #check for nonexsistant gradients
    c=0
    while c<len(grads):
      if grads[c]==None:
        #remove this trainable variable
        del grads[c]
        del myTrainableVariables[c]
      else:
        c+=1

    optimizer.apply_gradients(zip(grads,myTrainableVariables))

    return error


  def clear(self,exceptionNodes=[]):
    for nodeID in self.nodes.keys():
      if nodeID in exceptionNodes:
        continue
      else:
        self.nodes[nodeID].clear()

  def protectedClear(self,exceptionNodes=[]):
    for nodeID in self.nodes.keys():
      if nodeID in exceptionNodes:
        continue
      else:
        self.nodes[nodeID].protectedClear()
 

  def exportNetwork(self,networkPath:str,debug=True):
    from os import mkdir
    #make path
    try:
      mkdir(networkPath)
    except FileExistsError:
      pass
    except Exception as e:
      raise(invalidPath(networkPath))

    #write list of output nodes
    outNodeIDs=""
    for node in self.outputNodes:
      outNodeIDs+=str(node.ID)+"\n"
    outNodeIDs=outNodeIDs[:-1]#remove last \n
    with open(networkPath+"\\outputNodes.txt","w") as f:
      f.write(outNodeIDs)

    exported=[]#a list of IDs of nodes already exported
    #recursive function
    def exportFromNode(networkPath:str,node:Node,exported,debug):
      if node.ID in exported:
        return
      #else
      exported.append(node.ID)
      if debug:
        print(f"Exproting NODE{node.ID} ")
      node.exportNode(networkPath,f"NODE{node.ID}")
      #export connections
      for connectedNode in node.inputConnections:
        exportFromNode(networkPath,connectedNode,exported,debug)

    #recursive call starting from outputs
    for node in self.outputNodes:
      exportFromNode(networkPath,node,exported,debug)
 
  #made to be run on an empty network
  def importNetwork(self,networkPath:str):
    #clear stuff
    self.inputNodes=[]
    self.outputNodes=[]
    self.nodes={}

    from os import path

    #check if directory exists
    if not path.exists(networkPath):
      raise(missingDirectoryForImport(networkPath))

    #get list of output nodes
    outputNodeIDs=[]
    try:
      with open(networkPath+"\\outputNodes.txt","r") as f:
        rawLines=f.readlines()
        for line in rawLines:
          try:
            outputNodeIDs.append(int(line))
          except ValueError as e:
            raise(invalidDataInFile(networkPath+"\\outputNodes.txt","nodeID",line))
    except FileNotFoundError as e:
      raise(missingFileForImport(networkPath,"outputNodes.txt"))
     
    importedIDs=[]
    allNodes=[]
    #recursive function
    def importFromNode(networkPath:str,nodeID:int,importedIDs,allNodes) -> Node:
      if nodeID in importedIDs:
        return
      else:
        importedIDs.append(nodeID)

      accessPath=networkPath+"\\NODE"+str(nodeID)
      #get node type
      try:
        with open(accessPath+"\\type.txt","r") as f:
          name=f.readlines()[0]
      except FileNotFoundError as e:
        raise(missingFileForImport(accessPath,"type.txt"))
      
      nodeClass=NodeNameLookup.getNodeFromName(name)
      if nodeClass==None:
        #wrong name
        raise(invalidDataInFile(accessPath+"\\type.txt","name",name))
      myNode=nodeClass(ID=nodeID)
      allNodes.append(myNode)

      drop,connections=myNode.importNode(networkPath,f"NODE{nodeID}")
      nodesToConnect=[]
      for connection in connections:
        connectedNode=importFromNode(networkPath,connection,importedIDs,allNodes)
        nodesToConnect.append(connectedNode) 

      myNode.connect(nodesToConnect)

      return myNode
    
    #run the recursive function for all output nodes
    for nodeID in outputNodeIDs:
      self.outputNodes.append(importFromNode(networkPath,nodeID,importedIDs,allNodes))

    for node in allNodes:
      self.nodes[node.ID]=node
      if isinstance(node,InputNode):
        self.inputNodes.append(node)
    
    self.built=True