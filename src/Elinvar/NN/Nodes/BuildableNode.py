from .Node import Node 

class BuildableNode(Node):
  def __init__(self,name=None,protected=False):
    super().__init__(name=name,protected=protected)
    self.built=False

  #made to be overwritten
  def build(self):
    self.built=True