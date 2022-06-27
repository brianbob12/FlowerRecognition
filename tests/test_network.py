import sys
sys.path.append("../")

import Elinvar

def test_NodeIDsAreUnique():

  #setup neural net
  myNet=Elinvar.NN.Network()
  input1 = Elinvar.NN.Nodes.InputNode()
  input1.setup([256,256,3])
  conv1= Elinvar.NN.ConvolutionLayer() 
  conv1.newLayer(11,48,3,0)
  conv1.connect([input1])
  inst1=Elinvar.NN.InstanceNormalizationNode()
  inst1.newLayer(1,1)
  inst1.connect([conv1])
  pool1=Elinvar.NN.Nodes.MaxPoolingNode()
  pool1.newLayer(5,2)
  pool1.connect([inst1])
  conv2=Elinvar.NN.ConvolutionLayer()
  conv2.newLayer(11,48,3,0)
  conv2.connect([pool1])
  inst2=Elinvar.NN.InstanceNormalizationNode()
  inst2.newLayer(1,1)
  inst2.connect([conv2])
  pool2 = Elinvar.NN.MaxPoolingNode()
  pool2.newLayer(3,1)
  pool2.connect([inst2])
  flatten1=Elinvar.NN.Nodes.FlattenNode()
  flatten1.connect([pool2])
  dense1=Elinvar.NN.Nodes.DenseLayer()
  dense1.newLayer(128,"relu")
  dense1.connect([flatten1])
  dense2=Elinvar.NN.Nodes.DenseLayer()
  dense2.newLayer(64,"relu")
  dense2.connect([dense1])
  dense3=Elinvar.NN.Nodes.DenseLayer()
  dense3.newLayer(5,"sigmoid")
  dense3.connect([dense2])
  myNet.addInputNodes([input1])
  myNet.addNodes([conv1,inst1,pool1,conv2,inst2,pool2,flatten1,dense1,dense2])
  myNet.addOutputNodes([dense3])

  #confirm that node IDs are unique
  nodeIDs=[]
  for node in myNet.nodes.values():
    nodeIDs.append(node.id)

  assert len(nodeIDs)==len(set(nodeIDs))
