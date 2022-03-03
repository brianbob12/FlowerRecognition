from .Node import Node 

class BuildableNode(Node):
  def __init__(self):
    self.built=False

  #made to be overwritten
  def build(self):
    self.built=True