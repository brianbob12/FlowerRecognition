from random import randint
from NN import *

class Network:
  
  def __init__(self):
    self.nodes={}
    self.inputNodes=[]
    self.outputNodes=[]
    self.built=False

  def addNodes(self,nodes):
    for node in nodes:
      #check for ID collisions
      nodeID=node.ID
      changed=False
      while nodeID in self.nodes.keys():
        nodeID=randint(0,2**31)
        changed=True
      if changed:
        node.ID=nodeID
      self.nodes[nodeID]=node

  def addInputNodes(self,nodes):
    self.AddNodes(nodes)
    self.addInputNodes+=nodes

  def addOutputNodes(self,nodes):
    self.AddNodes(nodes)
    self.addOutputNodes+=nodes

  #private
  def buildNode(self,node):
    if isinstance(node,BuildableNode):
      if node.build:
        return
    for inputNode in node.inputConnections:
      self.buildNode(inputNode)

  def build(self):
    for outputNode in self.outputNodes:
      self.buildNode(outputNode)
    self.built=True

  def execute(self,inputs,requestedOutputs):
    #clear
    self.clear
    #deal with inputs
    for key in inputs.keys():
      self.nodes[key].onExecute=(lambda value:(lambda :value))(inputs[key])
    
    #produce outputs
    return([i.getValue() for i in requestedOutputs])

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