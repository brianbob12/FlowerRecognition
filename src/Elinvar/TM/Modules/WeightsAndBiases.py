from Elinvar.TM import TrainingEpisode
from Elinvar.NN.Nodes import NodeNameLookup
from .Module import Module


#User may not have dependency
import wandb

class WeightsAndBiases(Module):
  def __init__(self, project:str,entity:str):
    self.project:str=project
    self.entity:str=entity
    self.currentRun:wandb.Run=None

  def startOfEpisode(self, trainingEpisode: TrainingEpisode, episodeIndex: int):
    #produce layer makeup string
    layerMakeup=""
    for node in trainingEpisode.network.nodes.values():
      layerMakeup+=NodeNameLookup.getNameFromNode(node)+f"[{node.totalTrainableVariables}],"
    layerMakeup=layerMakeup[:-1]#remove last comma
    self.currentRun=wandb.init(
      config={
        "batchSize":trainingEpisode.batchSize,
        "crossValSetSeed":trainingEpisode.crossValSelectionSeed,
        "crossValSetSize":trainingEpisode.crossValSize,
        "layerMakeup":layerMakeup,
        "learningRate":trainingEpisode.trainingProtocol.learningRate,
        "numberOfNodes":trainingEpisode.network.nodes.__len__(),
        "totalTrainableVariables":trainingEpisode.network.getTotalTrainableVarialbes()
      },
      project=self.project,
      entity=self.entity,
      reinit=True,
      name=trainingEpisode.name
    )

  def endOfIteration(self, trainingEpisode: TrainingEpisode, index:int, trainingError: float, iterationTime: float):
    wandb.log({
      "index":index,
      "trainingerror":trainingError,
      "iterationtime":iterationTime
    })

  def endOfCrossVal(self, trainingEpisode: TrainingEpisode,index:int, crossValError: float):
    wandb.log({
      "index":index,
      "crossValError":crossValError
    })

  def endOfEpisode(self, trainingEpisode: TrainingEpisode,lastCrossValError:float):
    self.currentRun.summary["lastCrossValError"]=lastCrossValError
    self.currentRun.finish()