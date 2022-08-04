from copy import copy
import Elinvar
from Elinvar.NN.Nodes.DenseLayer import DenseLayer

#discriminator
def newDiscriminator():
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

  return myNet

def newGenerator():
  styleSize=10

  #function to create a convolution block
  def convBlock(
    inputNode:Elinvar.NN.Node,
    styleInput:Elinvar.NN.Node,
    filterSize:int,
    numberOfFilters:int
    ):

    #first convolution layer
    conv=Elinvar.NN.ConvolutionLayer()
    conv.newLayer(filterSize,numberOfFilters,1,0)
    conv.connect([inputNode])

    #create noise input
    noiseInput=Elinvar.NN.Nodes.InputNode()
    noiseInput.setup(copy(conv.outputShape))

    #create layer to add noise
    addLayer=Elinvar.NN.Nodes.ElementwiseAddition()    
    addLayer.connect([conv,noiseInput])

    #create a dense layer to connect style input to the correct size
    dense=Elinvar.NN.Nodes.DenseLayer()
    #each filter has a mean and standard deviation set by the dense layer
    dense.newLayer(numberOfFilters*2,"linear")
    dense.connect([styleInput])

    #create layer to add style
    ada=Elinvar.NN.Nodes.AdaINSetStyle()
    ada.connect([dense,addLayer])

    #return nodes of the block
    #format ([newInputNodes],[nodes],[outputNodes])
    return ([noiseInput],[conv,addLayer,dense],[ada])


  #important parameters

  myNet=Elinvar.NN.Network()

  #one latent image input, one noise input, and one style input

  latentInput=Elinvar.NN.Nodes.InputNode()
  latentInput.setup([256,256,3])

  styleInput=Elinvar.NN.Nodes.InputNode()
  styleInput.setup([5])

  noiseInputs=[]

  nodes=[]
  #nodes excludes input and output nodes

  preUpsample=[(11,48),(3,24)]
  postUpsample=[(3,12),(3,3)]

  previousOutput=latentInput
  for size,number in preUpsample:
    # add a convolution block
    newInputs,newNodes,newOutputs=convBlock(previousOutput,styleInput,size,number)
    noiseInputs.extend(newInputs)
    nodes.extend(newNodes)
    nodes.extend(newOutputs)
    previousOutput=newOutputs[0]

  # add upsampling layer
  upSample=Elinvar.NN.Nodes.UpsampleNode()
  upSample.new(300,300)
  upSample.connect([previousOutput])
  previousOutput=upSample

  for size,number in postUpsample:
    # add a convolution block
    newInputs,newNodes,newOutputs=convBlock(previousOutput,styleInput,size,number)
    noiseInputs.extend(newInputs)
    nodes.extend(newNodes)
    nodes.extend(newOutputs)
    previousOutput=newOutputs[0]

  myNet.addInputNodes([latentInput,styleInput]+noiseInputs)
  myNet.addNodes(nodes)
  myNet.addOutputNodes([previousOutput])

  return myNet
