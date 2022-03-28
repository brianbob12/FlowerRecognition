from random import randint
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

  #private
  def buildNode(self,node,seed):
    #first build connections
    for i,inputNode in enumerate(node.inputConnections):
      self.buildNode(inputNode,seed=seed+3*i)

    #then if buildable, build
    if isinstance(node,BuildableNode):
      if node.built:
        return
      else:
        node.build(seed=seed)
    
    

  def build(self,seed=None):
    if seed==None:
      seed=randint(0,2**31)
    for i,outputNode in enumerate(self.outputNodes):
      self.buildNode(outputNode,seed+i*2)
    self.built=True
    return seed

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