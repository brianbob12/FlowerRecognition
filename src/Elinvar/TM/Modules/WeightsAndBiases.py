from random import triangular
from Elinvar.TM import TrainingEpisode
from .Module import Module

#User may not have dependency
import wandb

class WeightsAndBiases(Module):
  def __init__(self, project:str,entity:str):
    self.project:str=project
    self.entity:str=entity
    self.currentRun:wandb.Run=None

  def startOfEpisode(self, trainingEpisode: TrainingEpisode, episodeIndex: int):
    self.currentRun=wandb.init(
      config={
        "batchSize":trainingEpisode.batchSize,
        "crossValSetSeed":trainingEpisode.crossValSelectionSeed,
        "crossValSetSize":trainingEpisode.crossValSize,
        "layerMakeup":"",
        "learningRate":trainingEpisode.trainingProtocol.learningRate,
        "numberOfNodes":trainingEpisode.network.nodes.__len__(),
        "totalTrainableVariables":trainingEpisode.network.getTotalTrainableVarialbes()
      },
      project=self.project,
      entity=self.entity,
      reinit=True
    )

  def endOfIteration(self, trainingEpisode: TrainingEpisode, index:int, trainingError: float, iterationTime: float):
    wandb.log({
      "index":index,
      "trainingError":trainingError,
      "iterationTime":iterationTime
    })

  def endOfCrossVal(self, trainingEpisode: TrainingEpisode,index:int, crossValError: float):
    return

  def endOfEpisode(self, trainingEpisode: TrainingEpisode,lastCrossValError:float):
    self.currentRun.finish()