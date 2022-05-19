from Elinvar.TM import TrainingEpisode
from .Condition import Condition

class ReachedIteration(Condition):
  __slots__=("target")

  def __init__(self,target:int):
    super().__init__()
    self.target:int=target

  def startOfEpisode(self, trainingEpisode: TrainingEpisode, episodeIndex: int):
    self.setValue(False)

  def endOfIteration(self, trainingEpisode: TrainingEpisode, index: int, trainingError: float, iterationTime: float):
    if(index>=self.target):
      self.setValue(True)