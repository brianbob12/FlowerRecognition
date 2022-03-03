
#class that holds nodes in the network
#these nodes may be junctions, layers, inputs or outputs
#each node has input connections to other nodes each connection has a shape

#each node has a stored value per execution
#the node can be cleared to clear this value thorugh the function clear
#a node can be perserved which disables the clear function

class Node:
  def __init__(self,name=None,protected=False):
    self.name=name
    self.inputConnections=[]#list of nodes
    self.outputShape=None#outputshape
    self.executionValue=None
    self.executed=False
    self.protected=protected
    self.value=None
    self.hasTrainableVariables=False

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
