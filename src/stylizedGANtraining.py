import Elinvar
import tensorflow as tf
import numpy as np
from PIL import Image

from dataRetrieval import getBatch, getFiles
from NetworkDesigns import newDiscriminator, newGenerator

#series A trains the discriminator, series B trains the generator

generatorNetwork=newGenerator()
discriminatorNetwork=newDiscriminator()

def trainGenerator(index)->Elinvar.TM.TrainingEpisode:
  name=f"GENERATOR{index}"
  learningRate=3e-6

  myErrorFunction=Elinvar.NN.ErrorFunctions.SoftmaxCrossEntropyWithLogits()
  myTrainingProtocol=Elinvar.NN.TrainingProtocols.DiscriminatorNetworkProtocol(
    learningRate,
    tf.keras.optimizers.Adam,
    [generatorNetwork.outputNodes[0]],
    discriminatorNetwork,
    [discriminatorNetwork.outputNodes[0]],
    {generatorNetwork.outputNodes[0]:discriminatorNetwork.inputNodes[0]},
    myErrorFunction
  )

  #TODO dataset

  te=Elinvar.TM.TrainingEpisode(name,generatorNetwork,myTrainingProtocol)

  return te

def trainDiscriminator(index)->Elinvar.TM.TrainingEpisode:
  name=f"DISCRIMINATOR{index}"
  learningRate=3e-6

def train(index)-> Elinvar.TM.TrainingEpisode:
  if index%2==0:
    return trainGenerator(index)
  else:
    return trainDiscriminator(index)